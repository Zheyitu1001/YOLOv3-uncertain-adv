import torch
import torch.nn as nn
import math

import numpy as np


class my_nll_loss(nn.Module):
    def __init___(self):
        super(my_nll_loss, self).__init__()

    #gt_pred:(batch_size, num_anchors, grid_size, grid_size, 1), xywh after rescale
    #mu: (batch_size, num_anchors, grid_size, grid_size, 1)
    #sigma: (batch_size, num_anchors, grid_size, grid_size, 1)
    #gt_ratio_wh: (batch_size, num_anchors, grid_size, grid_size, 1), 没有经过rescale的ground truth所对应的最佳anchor box
    #              在原始image中的比例
    def forward(self, gt_pred_t, mu, sigma, gt_ratio_wh):
        sigma_2 = torch.pow((sigma + 0.3), 2)
        z = torch.pow((2 * torch.tensor(np.pi) * sigma_2), 0.5)
        gt_mu_2 = torch.pow((gt_pred_t - mu), 2)
        prob_density = torch.exp(-0.5 * gt_mu_2 / (2 * sigma_2)) / z
        nll_loss = -torch.log(prob_density + 1e-7)
        nll_loss_scale = 0.5 * gt_ratio_wh * nll_loss
        return nll_loss_scale.mean()


