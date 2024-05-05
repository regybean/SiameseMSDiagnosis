#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasetz
from torchvision.transforms import Resize, ToTensor, Compose, Grayscale, ColorJitter,RandomAffine,RandomResizedCrop,RandomHorizontalFlip,Normalize,CenterCrop,ToPILImage
import argparse, random
from sklearn.model_selection import KFold
import logging

import networks
import datasets
from metrics import AccumulatedAccuracyMetric, F1Metric, AUCMetric
from transformations import SpeckleNoise, GaussianNoise, Rotate, BrightnessContrast


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cuda = torch.cuda.is_available()
print(device, torch.cuda.is_available())
torch.cuda.empty_cache()
logging.getLogger().setLevel(logging.ERROR)


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str, help='file name of the model to test in \models')
parser.add_argument('--dataset_path', default="./OCTdata/test/", type=str, help='path to the testing dataset')
parser.add_argument('--batch_size', default=16, type=int, help='It is the size of your batch.')
args = parser.parse_args()

# Loads the given model 
model_args = torch.load(args.model_path)
seed = model_args['seed']

# Fixes all random proccess
def reseed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms= True
    g = torch.Generator()
    g.manual_seed(seed)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def test(model, device, test_loader, loss_function, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        useTarget = True
        for batch_idx, (data,target) in enumerate(test_loader):
            data = tuple(d.to(device) for d in data)
            if len(data) == 3:
                useTarget = False

            outputs = model(*data)
                
            if target is not None:
                target = target.to(device)

            
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
                
            loss_inputs = outputs
            
            if useTarget:
                target = (target.float(),)
                loss_inputs += target
                
            pred = torch.where(outputs[0] > 0.5, 1, 0)
            print(f'\nPredictions = {pred},\nTargets = {target[0]}')      
            
            # Calculate the loss
            loss_outputs = loss_function(*loss_inputs)
            loss = loss_outputs
            val_loss += loss.item()
            
            for metric in metrics:
                metric(outputs, target, loss_outputs)

    
    return val_loss, metrics

# gets loaders 
def setup(args, dataset, corrupt = {'amountSpeckle':0.6,
             'amountGaussian':0.16,
             'degrees':-9,
             'amountBrightness':0.94,
             'amountContrast':1.18,
            }):
    
    reseed()
    test_folder = datasetz.ImageFolder(root=args.dataset_path)

    # Data preparation
    test_transform_fixed = [Resize(300),Grayscale(),ToTensor(),Normalize([0.5241], [0.1122])]
    
     # Simulated set data preparation
    transform_corrupt = [SpeckleNoise(corrupt['amountSpeckle'],True),
                         GaussianNoise(corrupt['amountGaussian'],True),
                         GaussianNoise(0.01,False,True),
                         Rotate(angle=corrupt['degrees']),
                         BrightnessContrast(brightness=corrupt['amountBrightness'], contrast=corrupt['amountContrast']),
                         ColorJitter(brightness=0.02),
                         Resize(300),
                         Grayscale(),
                         ToTensor(),
                         Normalize([0.5241], [0.1122]),
                        ]
    
    # Collects the different metrics to use
    metrics =[AccumulatedAccuracyMetric(), F1Metric(), AUCMetric()]
    
    # Normal run through
    
    # Forms the dataset and loaders with their augments
    test_dataset = getattr(datasets, dataset)(test_folder,[],test_transform_fixed,0.5,False)
    corrupt_dataset = getattr(datasets, dataset)(test_folder,[],transform_corrupt,0,False)
    
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False, worker_init_fn=seed_worker)
    corrupt_loader = DataLoader(corrupt_dataset,batch_size=args.batch_size,shuffle=False, worker_init_fn=seed_worker)
        
    
    return test_loader, corrupt_loader, metrics


# Model loading
if model_args['model_type'] == "sharedContrastive":
    model = getattr(networks,model_args['model_type'])(getattr(networks, 'EmbeddingNet')(model_args['arch'],model_args['frzlayers'],model_args['pretrained']))
elif model_args['model_type'] == "unsharedContrastive":
    model = getattr(networks,model_args['model_type'])(getattr(networks, 'EmbeddingNet')(model_args['arch'],model_args['frzlayers'],model_args['pretrained']),getattr(networks, 'EmbeddingNet')(model_args['arch'],model_args['frzlayers'],model_args['pretrained']))
else:
    model = getattr(networks,model_args['model_type'])(model_args['arch'],model_args['frzlayers'],model_args['pretrained'])   

model.load_state_dict(model_args['model_state_dict'])
loss_function = model_args['loss']
dataset = model_args['dataset']

test_loader, corrupt_loader, metrics = setup(args, dataset)
model = model.to(device)
metricList2 = []
metricList = []

# Tests the model on the test set
test_loss, metrics = test(model, device, test_loader, loss_function, metrics)
test_loss /= len(test_loader)
message = '\nTest set: Loss: {:.4f}'.format(test_loss)
for i, metric in enumerate(metrics):
    message += '  {}: {}'.format(metric.name(),metric.value())
    metricList.append(metric.value()) 


# Tests the model on the simulated set    
corrupt_loss, metrics= test(model, device, corrupt_loader, loss_function, metrics)
corrupt_loss /= len(corrupt_loader)
message += '\nSimulated set: Loss: {:.4f}'.format(corrupt_loss)
for i, metric in enumerate(metrics):
    message += '  {}: {}'.format(metric.name(), metric.value())
    metricList2.append(metric.value())

print(message)




