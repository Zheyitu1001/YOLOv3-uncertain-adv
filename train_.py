# ./train.py

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args([])
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 加载各个类的名字
classes = load_classes(opt.class_path)

# 加载数据集相关配置(主要是路径)
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# 获取模型超参数
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# 初始化创建模型结构
model = Darknet(opt.model_config_path)

# 随机初始化权重, weights_init_normal是定义在utils.py文件中函数, 会对模型进行高斯随机初始化
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train() # 将模型置于训练模式

# ListDataset是用于训练时使用的数据集类, 它会返回以下三个变量:
# 图片路径(str), 图片(3,416,416), 以及图片的box标签信息(50,5)
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 下式的lambda函数等价于: Adam(p for p in model.parameters() if p.requires_grad== True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        # imgs: [16, 3, 416, 416]
        # targets: [16, 50, 5]
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        # 清空优化器中的缓存梯度
        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward() # 执行反向传播算法
        optimizer.step() # 根据梯度对参数进行更新

        # 打印当前训练状态的各项损失值
        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        # 记录当前处理过的图片的总数
        model.seen += imgs.size(0) # 16
    if epoch % opt.checkpoint_interval == 0:
        # 调用 ./models.py 文件中的 save_weights 函数, 将训练好的参数权重进行存储
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
