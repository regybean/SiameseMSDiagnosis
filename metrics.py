#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import numpy as np
from torcheval.metrics import BinaryF1Score, BinaryAUROC

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError
        
class AUCMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.metric = BinaryAUROC()

    def __call__(self, outputs, target, loss):
        # They must change depending on the network which we use 
        if len(outputs) == 2:
            distances = (outputs[1] - outputs[0]).pow(2).sum(1).pow(0.5)
            self.metric.update(distances.cpu(), target[0].cpu())
            return self.value()
        else:
            self.metric.update(outputs[0].cpu(), target[0].cpu())
            return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.metric.reset()

    def value(self):
        return self.metric.compute().item()

    def name(self):
        return 'AUC'
    
class F1Metric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.metric = BinaryF1Score()

    def __call__(self, outputs, target, loss):
        if len(outputs) == 2:
            distances = (outputs[1] - outputs[0]).pow(2).sum(1).pow(0.5)
            pred = torch.where(distances > 0.5, 1, 0).cpu()
            self.metric.update(pred, target[0].cpu())
            return self.value()
        else:
            pred = torch.where(outputs[0] > 0.5, 1, 0).cpu()
            self.metric.update(pred, target[0].cpu())
            return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.metric.reset()

    def value(self):
        return self.metric.compute().item()

    def name(self):
        return 'F1score'
    
class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """
    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        # for triplet models
        if len(outputs) == 3:
            anchor, positive, negative = outputs
            distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
            distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
            self.correct += torch.gt(distance_negative - distance_positive, 0).sum().item()
            self.total += anchor.size(0) 
            return self.value()
        
        elif len(outputs) == 2:
            distances = (outputs[1] - outputs[0]).pow(2).sum(1).pow(0.5)
            pred = torch.where(distances > 0.5, 1, 0)
            self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
            self.total += target[0].size(0)
            return self.value()
        else:

            pred = torch.where(outputs[0] > 0.5, 1, 0)  # get the index of the max log-probability
            self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()

            self.total += target[0].size(0)

            return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        #print(self.correct, "/", self.total,"=", 100 * float(self.correct) / self.total)
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'

