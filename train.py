#!/usr/bin/env python
# coding: utf-8

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.datasets as datasetz
from torchvision.transforms import Resize, ToTensor, Compose, Grayscale, ColorJitter,RandomAffine,RandomResizedCrop,RandomHorizontalFlip,Normalize,CenterCrop,ToPILImage
from torch import optim
import argparse, random
import time 

import losses
import networks
import datasets
from metrics import AccumulatedAccuracyMetric, F1Metric, AUCMetric
from transformations import SpeckleNoise, GaussianNoise

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cuda = torch.cuda.is_available()
print(device, torch.cuda.is_available())

torch.cuda.empty_cache()

import logging
logging.getLogger().setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', default='SiameseOCTDataset', choices=['SiameseOCTDataset','SingularConcatOCTDataset','similaritySiameseOCTDataset','symmetrySiameseOCTDataset','symmetrySingularConcatOCTDataset','similaritySingularConcatOCTDataset'], type=str, help='type of dataset required from dataset.py')
parser.add_argument('--model', default='unsharedAggregated', type=str, choices=['sharedAggregated', 'unsharedAggregated','baseline','sharedContrastive','unsharedContrastive'], help='name of the model type to be trained')
parser.add_argument('--dataset_path', default="./OCTdata/train/", type=str, help='path to the training dataset')
parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet50', 'efficientnet_v2_s','densenet121','vgg11','alexnet','mobilenet_v3_small'], help='architecture of the model you wish to use - only compatable with Aggregated models')
parser.add_argument('--batch_size', default=16, type=int, help='It is the size of your batch.')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers used in the dataloader.')
parser.add_argument('--num_epochs', default=20, type=int, help='number of training epochs to run')
parser.add_argument('--pretrained', default=True, type=bool, help='specify whether the model is pretrained')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'RMSprop', 'SGD'], help='decide the optimizer to use from torch.optim')
parser.add_argument('--loss', default='BCELoss',choices=['BCELoss','DistanceContrastiveLoss','ContrastiveLoss'], type=str, help='loss function - contrastive models can only use contrastive loss')
parser.add_argument('--affine', default=False, type=bool, help='applies a random affine transformation')
parser.add_argument('--jitter', default=False, type=bool, help='applies a random colour and contrast jitter')
parser.add_argument('--flip', default=False, type=bool, help='applies a random flip to the images')
parser.add_argument('--gaussian', default=False, type=bool, help='adds random gaussian noise to the images')
parser.add_argument('--speckle', default=False, type=bool, help='adds random speckle noise to the images')
parser.add_argument('--normalize', default=True, type=bool, help='normalizes tensor values')
parser.add_argument('--frzlayers', default=None, type=int, help='number of layers to freeze using transfer learning')
parser.add_argument('--seed', default=14, type=int, help='choose the random seed for fixed testing')
parser.add_argument('--save', default=True, type=bool, help='whether to save the model after training')
args = parser.parse_args()

# Fixes all random proccess
def reseed():
    seed = args.seed
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



def train(model, train_loader, optimizer, loss_function, device, metrics):
    for metric in metrics:
        metric.reset()
    total_loss = 0
    losses = []
    useTarget = True
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data = tuple(d.to(device) for d in data)
            
        if len(data) == 3:
            useTarget = False
        outputs = model(*data)
                
        
        if useTarget:
            target = target.to(device)

        optimizer.zero_grad()
        
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)    
            
        loss_inputs = outputs
        
        if useTarget:
            target = (target.float(),)
            loss_inputs += target  

        loss_outputs =  loss_function(*loss_inputs)
        
        # Calculates the loss and backpropagates
        loss = loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        for metric in metrics:
            met = metric(outputs, target, loss_outputs)
            
        # Shows the progress 
        if batch_idx % 10 == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '  {}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []
        
    return total_loss, metrics


def setup(args):
    
    reseed()
    dataset = args.dataset_type
    train_folder = datasetz.ImageFolder(root=args.dataset_path)
    # Construct augmentation list from given params

    train_transform_affine = []
    train_transform_fixed= []
    arch = getattr(torchvision.models, args.arch)
    if args.speckle:
        train_transform_fixed += [SpeckleNoise(0.2)]
        
    if args.gaussian:
        train_transform_fixed += [GaussianNoise(0.2)]
        
    if args.affine:
        train_transform_affine += [RandomAffine(degrees=(-20,20),translate = (0.015625,0.015625),scale=(0.9,1.1),shear=(-10,10,-10,10))]
        
    if args.jitter:
        train_transform_affine += [ColorJitter(brightness=0.15, contrast=0.15)]
        
    if args.flip:
        train_transform_affine += [RandomHorizontalFlip(1)]
    
    # Data preparation
    train_transform_fixed += [Resize(300),Grayscale(),ToTensor(),Normalize([0.5241], [0.1122])]
    
    # Initialises the model based on its type
    if args.model == 'sharedAggregated':
        try:
            loss_function = getattr(nn, args.loss)()
        except:
            loss_function = getattr(losses, args.loss)()
        model = getattr(networks, args.model)(arch, args.frzlayers, args.pretrained)

    elif args.model == 'unsharedAggregated':
        # hack to get proper LR pairs
        dataset = "halved" + args.dataset_type
        try:
            loss_function = getattr(nn, args.loss)()
        except:
            loss_function = getattr(losses, args.loss)()
        model = getattr(networks, args.model)(arch, args.frzlayers, args.pretrained) 
        
    elif args.model == 'baseline':
        try:
            loss_function = getattr(nn, args.loss)()
        except:
            loss_function = getattr(losses, args.loss)()
        model = getattr(networks, args.model)(arch, args.frzlayers, args.pretrained)

    elif args.model == 'sharedContrastive':
        embedding_net = getattr(networks, 'EmbeddingNet')(arch, args.frzlayers, args.pretrained)
        try:
            loss_function = getattr(nn, args.loss)()
        except:
            loss_function = getattr(losses, args.loss)()
        model = getattr(networks, args.model)(embedding_net)
        
    elif args.model == 'unsharedContrastive':
        # hack to get proper LR pairs
        dataset = "halved" + args.dataset_type
        embedding_net = getattr(networks, 'EmbeddingNet')(arch, args.frzlayers, args.pretrained)
        embedding_net2 = getattr(networks, 'EmbeddingNet')(arch, args.frzlayers, args.pretrained)
        try:
            loss_function = getattr(nn, args.loss)()
        except:
            loss_function = getattr(losses, args.loss)()
        model = getattr(networks, args.model)(embedding_net,embedding_net2)
    else:
        print("wrong")
    
    # Collects the different metrics
    metrics =[AccumulatedAccuracyMetric(), F1Metric(), AUCMetric()]
    
    # Forms the dataset and loaders with their augments
    train_dataset = getattr(datasets, dataset)(train_folder,train_transform_affine,train_transform_fixed,0.5,True) 
        
    train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            worker_init_fn=seed_worker,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size)
        
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr= args.lr)
    scheduler = StepLR(optimizer,step_size= 8, gamma=0.1)
    

    return model, train_loader, optimizer, loss_function, scheduler, device, metrics, arch, dataset

# Training loop 
model, train_loader, optimizer, loss_function, scheduler, device, metrics, arch, dataset = setup(args)
writer = SummaryWriter()
model = model.to(device)
List = []
startt_time = time.time()
for epoch in range(1, args.num_epochs):
    start_time = time.time()

    # Trains the model
    train_loss, metrics = train(model, train_loader, optimizer, loss_function, device, metrics)
    train_loss /= len(train_loader)
    message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, args.num_epochs, train_loss)
    for i, metric in enumerate(metrics):
        if i ==0:
            List.append(metric.value())
        message += '  {}: {}'.format(metric.name(),metric.value())
        writer.add_scalar("TrainAccuracy", metric.value(), epoch)
        
    print(message)

    print("time taken:",time.time() - start_time)
    writer.add_scalar("TrainLoss", train_loss, epoch)
    scheduler.step()
 
# Saves the model    
if args.save:
    name = "models/model"+str(time.time())+".pt"
    torch.save({
            'arch': arch,
            'frzlayers':args.frzlayers,
            'pretrained':args.pretrained,
            'loss':loss_function,
            'seed':args.seed,
            'dataset':dataset,
            'model_type':args.model,
            'model_state_dict': model.state_dict()
            }, name)
    print("model saved as", name)
print("time taken is,", time.time() - startt_time)
