import torch
import torch.nn as nn

class vat_loss(nn.Module):
    def __init__(self, xi=1e-6, eps=2.5, num_iters=1):
        super(vat_loss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.iter = num_iters

    def forward(self, outputs, ul_x, ul_y):
        return None


