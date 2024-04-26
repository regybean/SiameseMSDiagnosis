#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
    # Require two inputs here 
    def forward(self, output1, output2, target, size_average=True):
        euc_dist = F.pairwise_distance(output1, output2)
   
        loss = torch.mean((1-target) * torch.pow(euc_dist, 2) +
          (target) * torch.pow(torch.clamp(self.margin - euc_dist, min=0.0), 2))

        return loss

     
class DistanceContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(DistanceContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
    # Require one distance metric instead
    def forward(self, dist, label):
        label = torch.tensor([0 if l == 1 else 1 for l in label])
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        label = label.to(device)

        loss = torch.mean(1/2*(label) * torch.pow(dist, 2) +
                                      1/2*(1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        

        return loss