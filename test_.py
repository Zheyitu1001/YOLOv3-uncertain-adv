# ./test.py

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args([])
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

# 获取数据集配置(路径)
data_config = parse_data_config(opt.data_config_path)
test_path = data_config["valid"]
num_classes = int(data_config["classes"])

# 初始化网络模型结构
model = Darknet(opt.model_config_path)

# 调用 ./models.py 文件中的 load_weights 函数加载模型的预训练权重
model.load_weights(opt.weights_path)

if cuda:
    model = model.cuda()

model.eval() # 将模型置于推演模式eval

# 获取数据集加载器, 这里需要根据数据的标签计算准确率, 因此需要使用ListDataset
dataset = ListDataset(test_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Compute mAP...")

all_detections = []
all_annotations = []

for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    imgs = Variable(imgs.type(Tensor))

    with torch.no_grad(): # 禁止计算梯度, 加快模型运算速度
        outputs = model(imgs)
        # 对计算结果执行 NMS 算法
        # outputs的shape为:[batch_size, m, 7]
        outputs = non_max_suppression(outputs, 80, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    for output, annotations in zip(outputs, targets): #targets的shape为:[batch_size, n, 5]
        # 根据类别的数量创建占位空间, all_detections为一个列表, 列表中只有一个元素,
        # 该元素还是一个列表, 该列表中有80个np元素
        all_detections.append([np.array([]) for _ in range(num_classes)])

        if output is not None:
            # 获取预测结果的相应值
            pred_boxes = output[:, :5].cpu().numpy() # 坐标和包含物体的概率obj_conf
            scores = output[:, 4].cpu().numpy() # 置信度
            pred_labels = output[:, -1].cput().numpy() # 类别编号

            # 按照置信度对预测的box进行排序
            sort_i = np.argsort(scores)
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]

            for label in range(num_classes):
                # all_detections是只有一个元素的列表, 因此这里用-1,
                # 获取所有预测类别为label的预测box, 可以将all_detections的shape看作为[1,1,80]
                all_detections[-1][label] = pred_boxes[pred_labels == label]

        # [1,1,80]
        all_annotations.append([np.array([]) for _ in range(num_classes)])

        if any(annotations[:, -1] > 0):
            annotations_labels = annotations[annotations[:, -1] > 0, 0].numpy() # 获取类别编号
            _annotation_boxes = annotations[annotations[:, -1] > 0, 1:].numpy() # 获取box坐标

            # 将box的格式转换成x1,y1,x2,y2的形式, 同时将图片放缩至opt.img_size大小
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
            annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
            # 因为原始的标签数据是以小数形式存储的, 所以可以直接利用乘法进行放缩
            annotation_boxes *= opt.img_size

            for label in range(num_classes):
                all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

# 以字典形式记录每一类的mAP值
average_precisions = {}
for label in range(num_classes):
    true_positives = []
    scores = []
    num_annotations = 0

    for i in tqdm.tqdm(range(len(all_annotations)), desc="Computing AP for class '{}'".format(label)):

        # 获取同类的预测结果和标签信息, i代表当前图片在batch中的位置
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]

        num_annotations += annotations.shape[0]
        detected_annotations = []

        for *bbox, score in detections:
            scores.append(score)

            if annotations.shape[0] == 0:
                true_positives.addpend(0) # 当前box并非真正例
                continue

            # 利用./utils/utils.py文件中的bbox_iou_numpy函数获取交并比矩阵(都是同类的box)
            overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1) # 获取最大交并比的下标
            max_overlap = overlaps[0, assigned_annotation] # 获取最大交并比

            if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)

    # 如果当前类没有出现在该图片中, 在当前类的 AP 为 0
    if num_annotations == 0:
        average_precisions[label] = 0
        continue

    true_positives = np.array(true_positives) # 将列表转化成numpy数组
    false_positives = np.ones_like(true_positives) - true_positives

    #按照socre进行排序
    indices = np.argsort(-np.array(scores))
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # 统计假正例和真正例
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # 计算召回率和准确率
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # 调用utils.py文件中的compute_ap函数计算average precision
    average_precision = compute_ap(recall, precision)
    average_precisions[label] = average_precision

print("Average Precisions:")
for c, ap in average_precisions.items():
    print("+ Class '{}' - AP: {}".format(c, ap))

mAP = np.mean(list(average_precisions.values()))
print("mAP: {}".format(mAP))
