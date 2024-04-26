#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import random
import torch
from torchvision import transforms 
from skimage.util import random_noise

seed = 14
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms= True

# For the shared models
class SiameseOCTDataset(Dataset):

    def __init__(self, data, transform_affine=[], transform_fixed=[],transform_prob = 0.5, train=True):
        
        self.data = data
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob
        self.train = train
        
        if self.train:
            self.train_labels = [d[1] for d in self.data.imgs]
            self.train_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.train_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}
            
        # generate fixed pairs for testing at initialisation
        else:
            self.test_labels = [d[1] for d in self.data.imgs]
            self.test_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.test_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)
            self.pairs = []
            for i in range(len(self.test_data)):
                if i %2 == 0:
                    pair = [i,i+1]
                else:
                    pair = [i,i-1]
                    
                self.pairs.append(pair)
     
            self.test_pairs = self.pairs
            

    def __len__(self):
        return len(self.data.imgs)

    def __getitem__(self, index):
        # We get two retinas either LR or RL
        if self.train:
            img0, label = self.train_data[index], self.train_labels[index]
            index = self.train_data.index(img0)
            
            if index %2 == 0:
                positive_index = index + 1
            else:
                positive_index = index -1

            img1 = self.train_data[positive_index]
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        else:
            label = self.test_labels[index]
            img0 = self.test_data[self.test_pairs[index][0]]
            img1 = self.test_data[self.test_pairs[index][1]]
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        # Apply the transformations at a probability of 50%
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                  
        for transform in self.transform_fixed:
            img0 = transform(img0)
            img1 = transform(img1)

        return (img0,img1),label
        
        
    def getData(self):
        return self.data

# For the unshared models
class halvedSiameseOCTDataset(Dataset):

    def __init__(self, data, transform_affine=[], transform_fixed=[],transform_prob = 0.5, train=True):
        
        self.data = data
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob
        self.train = train
        
        if self.train:
            self.train_data = [d[0] for d in self.data.imgs]
            self.train_data2 = [d[0] for i,d in enumerate(self.data.imgs) if not i%2]
            self.train_labels2 = [d[1] for i,d in enumerate(self.data.imgs) if not i%2]
        else:
            self.test_labels = [d[1] for d in self.data.imgs]
            self.test_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.test_labels))


            random_state = np.random.RandomState(29)
            self.pairs = []
            for i in range(len(self.test_data)):
                if i %2 == 0:
                    pair = [i,i+1]  
                    self.pairs.append(pair)
     
            self.test_pairs = self.pairs
            

    def __len__(self):
        return int(len(self.data.imgs)/2)

    def __getitem__(self, index):
        # gets only LR pairs
        if self.train:
            img0, label = self.train_data2[index], self.train_labels2[index]
            index = self.train_data.index(img0)
            img1 = self.train_data[index + 1]
            
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        else:
            img0 = self.test_data[self.test_pairs[index][0]]
            img1 = self.test_data[self.test_pairs[index][1]]
            index = self.test_data.index(img0)
            label = self.test_labels[index]
            
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                  
        for transform in self.transform_fixed:
            img0 = transform(img0)
            img1 = transform(img1)

        return (img0,img1),label
        
        
    def getData(self):
        return self.data

# For the baseline
class SingularConcatOCTDataset(Dataset):

    def __init__(self, data, transform_affine=[], transform_fixed=[],transform_prob = 0.5, train=True):
        
        self.data = data
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob
        self.train = train
        
        if self.train:
            self.train_labels = [d[1] for d in self.data.imgs]
            self.train_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.train_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            self.test_labels = [d[1] for d in self.data.imgs]
            self.test_data = [d[0] for d in self.data.imgs]
            # generate fixed pairs for testing
            self.labels_set = set(np.asarray(self.test_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)
            self.pairs = []
            for i in range(len(self.test_data)):
                if i %2 == 0:
                    pair = [i,i+1]
                else:
                    pair = [i,i-1]
                    
                self.pairs.append(pair)
     
            self.test_pairs = self.pairs
            

    def __len__(self):
        return len(self.data.imgs)

    def __getitem__(self, index):
        # get 2 items from random shuffle of the pairs of LR
        if self.train:
            img0, label = self.train_data[index], self.train_labels[index]
            #print(self.train_data)
            index = self.train_data.index(img0)
            
            if index %2 == 0:
                positive_index = index + 1
            else:
                positive_index = index -1

            img1 = self.train_data[positive_index]
            
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        else:
            label = self.test_labels[index]
            img0 = self.test_data[self.test_pairs[index][0]]
            img1 = self.test_data[self.test_pairs[index][1]]
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                    
        for transform in self.transform_fixed[:-2]:
            img0 = transform(img0)
            img1 = transform(img1)
            
        img0_size = img0.size
        img1_size = img1.size
        
        # The images are concatenated here after the augmentations
        new_image = Image.new('L',(2*img0_size[0], img0_size[1]))
        new_image.paste(img0,(0,0))
        new_image.paste(img1,(img0_size[0],0))
        
        for transform in self.transform_fixed[-2:]:
            new_image = transform(new_image)
            
        return (new_image,),label
        
        
    def getData(self):
        return self.data
    
class halvedsymmetrySiameseOCTDataset(Dataset):

    def __init__(self, data, transform_affine=[], transform_fixed=[],transform_prob = 0.5, train=True):
        
        self.data = data
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob
        self.train = train
        
        if self.train:
            self.train_data = [d[0] for d in self.data.imgs]
            self.train_data2 = [d[0] for i,d in enumerate(self.data.imgs) if not i%2]
            self.train_labels2 = [d[1] for i,d in enumerate(self.data.imgs) if not i%2]
        else:
            self.test_labels = [d[1] for d in self.data.imgs]
            self.test_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.test_labels))


            random_state = np.random.RandomState(29)
            self.pairs = []
            for i in range(len(self.test_data)):
                if i %2 == 0:
                    pair = [i,i+1]  
                    self.pairs.append(pair)
     
            self.test_pairs = self.pairs
            

    def __len__(self):
        return int(len(self.data.imgs)/2)

    def __getitem__(self, index):
        # get 2 items from random shuffle of the pairs of LR
        if self.train:
            img0, label = self.train_data2[index], self.train_labels2[index]
            index = self.train_data.index(img0)
            img1 = self.train_data[index + 1]
            
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)
            
        else:
            img0 = self.test_data[self.test_pairs[index][0]]
            img1 = self.test_data[self.test_pairs[index][1]]
            index = self.test_data.index(img0)
            label = self.test_labels[index]
            
            img0 = Image.open(img0)
            img1 = Image.open(img1)   
            
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                  
        for transform in self.transform_fixed:
            img0 = transform(img0)
            img1 = transform(img1)

        return (img0,img1),label
        
        
    def getData(self):
        return self.data
    
class halvedsimilaritySiameseOCTDataset(Dataset):

    def __init__(self, data, transform_affine=[], transform_fixed=[],transform_prob = 0.5, train=True):
        
        self.data = data
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob
        self.train = train
        
        if self.train:
            self.train_data = [d[0] for d in self.data.imgs]
            self.train_data2 = [d[0] for i,d in enumerate(self.data.imgs) if not i%2]
            self.train_labels2 = [d[1] for i,d in enumerate(self.data.imgs) if not i%2]
        else:
            self.test_labels = [d[1] for d in self.data.imgs]
            self.test_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.test_labels))


            random_state = np.random.RandomState(29)
            self.pairs = []
            for i in range(len(self.test_data)):
                if i %2 == 0:
                    pair = [i,i+1]  
                    self.pairs.append(pair)
     
            self.test_pairs = self.pairs
            

    def __len__(self):
        return int(len(self.data.imgs)/2)

    def __getitem__(self, index):
        # get 2 items from random shuffle of the pairs of LR
        if self.train:
            img0, label = self.train_data2[index], self.train_labels2[index]
            index = self.train_data.index(img0)
            img1 = self.train_data[index + 1]
            
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        else:
            img0 = self.test_data[self.test_pairs[index][0]]
            img1 = self.test_data[self.test_pairs[index][1]]
            index = self.test_data.index(img0)
            label = self.test_labels[index]
            
            img0 = Image.open(img0)
            img1 = Image.open(img1)
   
        img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)    
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                  
        for transform in self.transform_fixed:
            img0 = transform(img0)
            img1 = transform(img1)

        return (img0,img1),label
        
        
    def getData(self):
        return self.data
    
    
class similaritySiameseOCTDataset(Dataset):

    def __init__(self, data, transform_affine=[], transform_fixed=[],transform_prob = 0.5, train=True):
        
        self.data = data
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob
        self.train = train
        
        if self.train:
            self.train_labels = [d[1] for d in self.data.imgs]
            self.train_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.train_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            self.test_labels = [d[1] for d in self.data.imgs]
            self.test_data = [d[0] for d in self.data.imgs]
            # generate fixed pairs for testing
            self.labels_set = set(np.asarray(self.test_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)
            self.pairs = []
            for i in range(len(self.test_data)):
                if i %2 == 0:
                    pair = [i,i+1]
                else:
                    pair = [i,i-1]
                    
                self.pairs.append(pair)
     
            self.test_pairs = self.pairs
            

    def __len__(self):
        return len(self.data.imgs)

    def __getitem__(self, index):
        # get 2 items from random shuffle of the pairs of LR
        if self.train:
            img0, label = self.train_data[index], self.train_labels[index]
            #print(self.train_data)
            index = self.train_data.index(img0)
            
            if index %2 == 0:
                positive_index = index + 1
            else:
                positive_index = index -1

            img1 = self.train_data[positive_index]
            img0 = Image.open(img0)
            img1 = Image.open(img1)
        
            
        else:
            label = self.test_labels[index]
            img0 = self.test_data[self.test_pairs[index][0]]
            img1 = self.test_data[self.test_pairs[index][1]]
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)   
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                    
        for transform in self.transform_fixed:
            img0 = transform(img0)
            img1 = transform(img1)
            
        return (img0,img1),label
    
class symmetrySiameseOCTDataset(Dataset):

    def __init__(self, data, transform_affine=[], transform_fixed=[],transform_prob = 0.5, train=True):
        
        self.data = data
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob
        self.train = train
        
        if self.train:
            self.train_labels = [d[1] for d in self.data.imgs]
            self.train_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.train_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            self.test_labels = [d[1] for d in self.data.imgs]
            self.test_data = [d[0] for d in self.data.imgs]
            # generate fixed pairs for testing
            self.labels_set = set(np.asarray(self.test_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)
            self.pairs = []
            for i in range(len(self.test_data)):
                if i %2 == 0:
                    pair = [i,i+1]
                else:
                    pair = [i,i-1]
                    
                self.pairs.append(pair)
     
            self.test_pairs = self.pairs
            

    def __len__(self):
        return len(self.data.imgs)

    def __getitem__(self, index):
        # get 2 items from random shuffle of the pairs of LR
        if self.train:
            img0, label = self.train_data[index], self.train_labels[index]
            #print(self.train_data)
            index = self.train_data.index(img0)
            
            if index %2 == 0:
                positive_index = index + 1
            else:
                positive_index = index -1

            img1 = self.train_data[positive_index]
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)
            
        else:
            label = self.test_labels[index]
            img0 = self.test_data[self.test_pairs[index][0]]
            img1 = self.test_data[self.test_pairs[index][1]]
            img0 = Image.open(img0)
            img1 = Image.open(img1)
              
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                    
        for transform in self.transform_fixed:
            img0 = transform(img0)

            img1 = transform(img1)
            
        return (img0,img1),label
    
class symmetrySingularConcatOCTDataset(Dataset):

    def __init__(self, data, transform_affine=[], transform_fixed=[],transform_prob = 0.5, train=True):
        
        self.data = data
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob
        self.train = train
        
        if self.train:
            self.train_labels = [d[1] for d in self.data.imgs]
            self.train_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.train_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            self.test_labels = [d[1] for d in self.data.imgs]
            self.test_data = [d[0] for d in self.data.imgs]
            # generate fixed pairs for testing
            self.labels_set = set(np.asarray(self.test_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)
            self.pairs = []
            for i in range(len(self.test_data)):
                if i %2 == 0:
                    pair = [i,i+1]
                else:
                    pair = [i,i-1]
                    
                self.pairs.append(pair)
     
            self.test_pairs = self.pairs
            

    def __len__(self):
        return len(self.data.imgs)

    def __getitem__(self, index):
        # get 2 items from random shuffle of the pairs of LR
        if self.train:
            img0, label = self.train_data[index], self.train_labels[index]
            #print(self.train_data)
            index = self.train_data.index(img0)
            
            if index %2 == 0:
                positive_index = index + 1
            else:
                positive_index = index -1

            img1 = self.train_data[positive_index]
            
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        else:
            label = self.test_labels[index]
            img0 = self.test_data[self.test_pairs[index][0]]
            img1 = self.test_data[self.test_pairs[index][1]]
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)
        
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                    
        for transform in self.transform_fixed[:-2]:
            img0 = transform(img0)
            img1 = transform(img1)
            
        img0_size = img0.size
        img1_size = img1.size
        
        new_image = Image.new('L',(2*img0_size[0], img0_size[1]))
        new_image.paste(img0,(0,0))
        new_image.paste(img1,(img0_size[0],0))
        
        for transform in self.transform_fixed[-2:]:
            new_image = transform(new_image)
            
        return (new_image,),label
        
        
    def getData(self):
        return self.data
    
class similaritySingularConcatOCTDataset(Dataset):

    def __init__(self, data, transform_affine=[], transform_fixed=[],transform_prob = 0.5, train=True):
        
        self.data = data
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob
        self.train = train
        
        if self.train:
            self.train_labels = [d[1] for d in self.data.imgs]
            self.train_data = [d[0] for d in self.data.imgs]
            self.labels_set = set(np.asarray(self.train_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            self.test_labels = [d[1] for d in self.data.imgs]
            self.test_data = [d[0] for d in self.data.imgs]
            # generate fixed pairs for testing
            self.labels_set = set(np.asarray(self.test_labels))
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)
            self.pairs = []
            for i in range(len(self.test_data)):
                if i %2 == 0:
                    pair = [i,i+1]
                else:
                    pair = [i,i-1]
                    
                self.pairs.append(pair)
     
            self.test_pairs = self.pairs
            

    def __len__(self):
        return len(self.data.imgs)

    def __getitem__(self, index):
        # get 2 items from random shuffle of the pairs of LR
        if self.train:
            img0, label = self.train_data[index], self.train_labels[index]
            #print(self.train_data)
            index = self.train_data.index(img0)
            
            if index %2 == 0:
                positive_index = index + 1
            else:
                positive_index = index -1

            img1 = self.train_data[positive_index]
            
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        else:
            label = self.test_labels[index]
            img0 = self.test_data[self.test_pairs[index][0]]
            img1 = self.test_data[self.test_pairs[index][1]]
            img0 = Image.open(img0)
            img1 = Image.open(img1)
            
        img0 = img0.transpose(Image.FLIP_LEFT_RIGHT)
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                    
        for transform in self.transform_fixed[:-2]:
            img0 = transform(img0)
            img1 = transform(img1)
            
        img0_size = img0.size
        img1_size = img1.size
        
        new_image = Image.new('L',(2*img0_size[0], img0_size[1]))
        new_image.paste(img0,(0,0))
        new_image.paste(img1,(img0_size[0],0))
        
        for transform in self.transform_fixed[-2:]:
            new_image = transform(new_image)
            
        return (new_image,),label
        
        
    def getData(self):
        return self.data
    
class siameseWrapperDataset:
    def __init__(self, dataset, transform_affine=[], transform_fixed=[],transform_prob = 0.5):
        self.dataset = dataset
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob

    def __getitem__(self, index):
        (img0,img1), label = self.dataset[index]    
        
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                  
        for transform in self.transform_fixed:
            img0 = transform(img0)
            img1 = transform(img1)
            
        return (img0,img1),label

    def __len__(self):
        return len(self.dataset)
    
class singularWrapperDataset:
    def __init__(self, dataset, transform_affine=[], transform_fixed=[],transform_prob = 0.5):
        self.dataset = dataset
        self.transform_affine = transform_affine
        self.transform_fixed = transform_fixed
        self.transform_prob = transform_prob

    def __getitem__(self, index):
        (img0,img1), label = self.dataset[index]    
        
        for transform in self.transform_affine:
                if random.random() < self.transform_prob:
                    img0 = transform(img0)
                    img1 = transform(img1)
                  
        for transform in self.transform_fixed[:-2]:
            img0 = transform(img0)
            img1 = transform(img1)
            
        img0_size = img0.size
        img1_size = img1.size
        
        new_image = Image.new('L',(2*img0_size[0], img0_size[1]))
        new_image.paste(img0,(0,0))
        new_image.paste(img1,(img0_size[0],0))
        
        for transform in self.transform_fixed[-2:]:
            new_image = transform(new_image)
            
        return (new_image,),label

    def __len__(self):
        return len(self.dataset)