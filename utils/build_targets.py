import torch

from utils.utils import bbox_wh_iou, bbox_iou



def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """"""
    #pred_boxes: (batch_size, 3, grid_size, grid_size, 4)
    #pred_cls: (batch-size, 3, grid_size, grid_size, 80)
    #target: (batch_size, 50, 5), 50个label, 为中心坐标和长宽以及所属类别 ？？？ 6
    #anchors: (3, 2), 3个prior anchor每个有长宽两个信息
    #ignore_thresh: 0.5

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0) # batch_size
    nA = pred_boxes.size(1) # 3
    nC = pred_cls.size(-1)  # 80
    nG = pred_boxes.size(2) # grid_size

    # Output tensors
    #即第m个采样中的第k个anchor的第i行第j列的cell的预测量
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0) # (batch_size, 3, grid_size, grid_size), 0
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1) #
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG #gt
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    #每个gt为(1,4), anchor为(3,4), 返回的值为(3)
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    #返回iou最大值和其所对应的anchor的索引
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    # 负责预测gt的第b个sample中的第best_n个anchor在第ij个cell中的中心偏移量以及宽高比例
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf