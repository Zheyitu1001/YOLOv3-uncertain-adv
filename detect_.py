# ./detect.py

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args([])
print(opt)


# 指示当前cuda是否可用
cuda = torch.cuda.is_available() and opt.use_cuda

# 创建模型并加载权重
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

# 如果cuda可用, 则将model移至cuda
if cuda:
    model.cuda()

model.eval() # 将模型的状态置为eval状态(会改变月一些内置网络层的行为)

img_datasets = ImageFolder(opt.image_folder, img_size=opt.img_size)
dataloader = DataLoader(img_datasets,
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
                       )

# 调用utils/utils.py文件中的load_classes()函数加载类别的名字(['person','car',...,'toothbrush'])
classes = load_classes(opt.class_path)

Tensor = torch.cuda.FloatTensor if cuda else torch.FLoatTnesor

imgs = [] # 存储图片路径
img_detections = [] # 存储每张图片的检测结果

data_size = len(img_datasets) # 图片的数量
epoch_size = len(dataloader) # epoch的数量: data_size / batch_size

print ('\nPerforming object detection: {} images, {} epoches'.format(data_size, epoch_size))

prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # 配置输入
    input_imgs = Variable(input_imgs.type(Tensor)) # Tensor: FloatTensor

    # 获取预测结果
    with torch.no_grad():
        # detections的shape为: [1,10647,85], 其中, 1为batch_size
        # 因为对于尺寸为416的图片来说:(13*13+26*26+52*52) * 3 = 10647
        # 如果图片尺寸为608(必须为32的倍数), 那么就为:(19*19+38*38+76*76) * 3 = 22743
        detections = model(input_imgs)

        # nms: 对于每一类(不同类之间的box不执行nms), 先选出具有最大score的box,
        # 然后删除与该box交并比较大的同类box, 接着继续选下一个最大socre的box, 直至同类box为空
        # 注意yolo与faster rcnn在执行nms算法时的不同, 前者是在多类上执行的, 后者是在两类上执行的
        # 执行nms后, 这里的detections是一个列表, 列表中的每个元素代表着一张图片nms后的box集合
        # 每一张图片的shape为:[m, 7], m代表box的数量, 7代表:(x1,y1,x2,y2,obj_conf,class_conf,class_pred)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)


        #break

    # 记录当前时间
    current_time = time.time()
    # 计算detect花费的时间(一张图片)
    inference_time = datetime.timedelta(seconds=current_time - prev_time)

    # 更新prev_time
    prev_time = current_time

    # 打印到屏幕
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # 记录图片的路径和检测结果, 以便后面进行可视化
    imgs.extend(img_paths)
    img_detections.extend(detections)


# 检测完成后, 根据 imgs 和 img_detections 的值进行可视化(以.png图片形式存储在磁盘上)

# 设值边框的颜色
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print('\nSaving image:')

# 遍历所有的imgs 和 img_detections, 对检测结果进行可视化
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
    print ("(%d) Image: '%s'" % (img_i, path))

    # 创建plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img) # 将img添加到当前的plot中

    # 计算给当前图片添加的padding的像素数
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))

    # 获取移除padding之后的图片的宽和高, 注意这个宽和高不同图片的原始大小, 而是放缩后的大小(长边为opt.img_size)
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # 在图片上画相应的box的边框和标签
    if detections is not None:
        # 获取当前图片中出现过的标签
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels) # 获取出现过的标签的数量
        bbox_colors = random.sample(colors, n_cls_preds) # 为每个类别标签随机分配颜色

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # 输出当前box的标签和相应的概率
            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # 将坐标转换到原始图片上的像素坐标
            box_h = ((y2-y1) / unpad_h) * img.shape[0]
            box_w = ((x2-x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            # 获取当前类别的颜色
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

            # 创建矩形
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')

            # 将创建好的矩形添加到当前的plot中(会加载在图片的上面)
            ax.add_patch(bbox)
            # 添加标签
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top', bbox={'color':color, 'pad':0})

    # 将图片保存在磁盘上
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()
