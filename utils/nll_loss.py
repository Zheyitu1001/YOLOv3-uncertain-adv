import torch
import torch.nn as nn

import numpy as np

class my_nll_yolo(nn.Module):
    def __init__(self, mu, sigma, sigma_const=0.3):
        self.mu = mu
        self.sigma = sigma
        self.sigma_const = sigma_const
        self.pi = torch.tensor(np.pi)

    def forward(self, x):
        z = (2 * self.pi * (self.sigma + self.sigma_const) ** 2) ** 0.5
        prob_density = torch.exp(-0.5 * ((x - self.mu) ** 2) / ((self.sigma_const + self.sigma) ** 2)) / z
        nll = -torch.log(prob_density + 1e-7)
        return nll

def nll(x, mu, sigma, sigma_const=0.3):
    pi = torch.tensor(np.pi)
    z = (2 * pi * (sigma + sigma_const) ** 2) ** 0.5
    prob_density = torch.exp(-0.5 * ((x - mu) ** 2) / ((sigma + sigma_const) ** 2)) / z
    nll = -torch.log(prob_density + 1e-7)
    return nll