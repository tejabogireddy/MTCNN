import os
import numpy as np
import torch
from torch.autograd import Variable
from models.pnet import PNet
from models.rnet import RNet
from models.onet import ONet
from utils.image_loader import ImageLoader
import torchvision.transforms as transforms
from utils.losses import Loss
import datetime


def convert_image_to_tensor(image):
    transform = transforms.ToTensor()
    image = image.astype(np.float)
    return transform(image)


def compute_accuracy(prob_cls, gt_cls):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)
    mask = torch.ge(gt_cls, 0)
    valid_gt_cls = torch.masked_select(gt_cls, mask)
    valid_prob_cls = torch.masked_select(prob_cls, mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls, 0.6).float()
    right_ones = torch.eq(prob_ones, valid_gt_cls).float()
    return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))


def pnet_trainer(model_store_path, data, batch_size, learning_rate):
    network = PNet()
    loss = Loss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    adjusted_data = ImageLoader(data, 12, batch_size, shuffle=True)
    network.train()

    for epoch in range(10):
        adjusted_data.reset()

        for batch_index, (image, (label, bbox)) in enumerate(adjusted_data):
            im_tensor = [convert_image_to_tensor(image[i, :, :, :]) for i in range(image.shape[0])]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor).float()
            im_label = Variable(torch.from_numpy(label).float())
            im_bbox = Variable(torch.from_numpy(bbox).float())

            label_predictions, bbox_predictions = network(im_tensor)

            class_loss = loss.cls_loss(im_label, label_predictions)
            box_loss = loss.box_loss(im_label, im_bbox, bbox_predictions)

            total_loss = (class_loss * 1.0) + (box_loss * 0.5)

            if (batch_index % 100) == 0:
                accuracy = compute_accuracy(label_predictions, im_label)
                print("%s : Epoch: %d, Step: %d, accuracy: %s, detection: %s, bbox_loss: %s, total_loss: %s"
                      % (datetime.datetime.now(), epoch, batch_index, accuracy,
                      class_loss, box_loss, total_loss))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        torch.save(network.state_dict(), os.path.join(model_store_path, "pnet_epoch_%d.pt" % epoch))
        torch.save(network, os.path.join(model_store_path, "pnet_epoch_model_%d.pkl" % epoch))

def rnet_trainer(model_store_path, data, batch_size, learning_rate):
    network = RNet()
    loss = Loss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    adjusted_data = ImageLoader(data, 24, batch_size, shuffle=True)
    network.train()

    for epoch in range(10):
        adjusted_data.reset()

        for batch_index, (image, (label, bbox)) in enumerate(adjusted_data):
            try:
               im_tensor = [convert_image_to_tensor(image[i, :, :, :]) for i in range(image.shape[0])]
            except:
               continue

            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor).float()
            im_label = Variable(torch.from_numpy(label).float())
            im_bbox = Variable(torch.from_numpy(bbox).float())

            label_predictions, bbox_predictions = network(im_tensor)

            class_loss = loss.cls_loss(im_label, label_predictions)
            box_loss = loss.box_loss(im_label, im_bbox, bbox_predictions)

            total_loss = (class_loss * 1.0) + (box_loss * 0.5)

            if (batch_index % 100) == 0:
                accuracy = compute_accuracy(label_predictions, im_label)
                print("%s : Epoch: %d, Step: %d, accuracy: %s, detection: %s, bbox_loss: %s, total_loss: %s"
                      % (datetime.datetime.now(), epoch, batch_index, accuracy,
                      class_loss, box_loss, total_loss))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        torch.save(network.state_dict(), os.path.join(model_store_path, "rnet_epoch_%d.pt" % epoch))
        torch.save(network, os.path.join(model_store_path, "rnet_epoch_model_%d.pkl" % epoch))

def onet_trainer(model_store_path, data, batch_size, learning_rate):
    network = ONet()
    loss = Loss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    adjusted_data = ImageLoader(data, 48, batch_size, shuffle=True)
    network.train()

    for epoch in range(10):
        adjusted_data.reset()

        for batch_index, (image, (label, bbox)) in enumerate(adjusted_data):
            try:
               im_tensor = [convert_image_to_tensor(image[i, :, :, :]) for i in range(image.shape[0])]
            except:
               continue

            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor).float()
            im_label = Variable(torch.from_numpy(label).float())
            im_bbox = Variable(torch.from_numpy(bbox).float())

            label_predictions, bbox_predictions = network(im_tensor)

            class_loss = loss.cls_loss(im_label, label_predictions)
            box_loss = loss.box_loss(im_label, im_bbox, bbox_predictions)

            total_loss = (class_loss * 1.0) + (box_loss * 0.5)

            if (batch_index % 100) == 0:
                accuracy = compute_accuracy(label_predictions, im_label)
                print("%s : Epoch: %d, Step: %d, accuracy: %s, detection: %s, bbox_loss: %s, total_loss: %s"
                      % (datetime.datetime.now(), epoch, batch_index, accuracy,
                      class_loss, box_loss, total_loss))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        torch.save(network.state_dict(), os.path.join(model_store_path, "onet_epoch_%d.pt" % epoch))
        torch.save(network, os.path.join(model_store_path, "onet_epoch_model_%d.pkl" % epoch))
