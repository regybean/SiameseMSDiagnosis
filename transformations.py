import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import random

# Custom Augmentations
class SpeckleNoise(object):
    
    def __init__(self,amount, fixed=False, ranged=False):
        self.amount = amount
        self.gauss = np.random.normal(0,self.amount,(768,768))
        self.fixed = fixed
        self.ranged = ranged
        if self.ranged:
                self.amount = self.amount + random.uniform(0,0.02)

    def __call__(self, sample):
        img = np.array(sample)
        if self.fixed:
            self.gauss = self.gauss.reshape(img.shape[0],img.shape[1]).astype('uint8')
            noise = img + img * self.gauss
        else:
            gauss = np.random.normal(0,self.amount,img.size)
            gauss = gauss.reshape(img.shape[0],img.shape[1]).astype('uint8')
            noise = img + img * gauss
        
        return Image.fromarray(noise)
    
class GaussianNoise(object):
    
    def __init__(self,amount, fixed=False, ranged=False):
        self.amount = amount
        self.gauss = np.random.normal(0,self.amount,(768,768))
        self.fixed = fixed
        self.ranged = ranged
        if self.ranged:
                self.amount = self.amount + random.uniform(0,0.02)
    
    def __call__(self, sample):
        img = np.array(sample)
        if self.fixed:
            self.gauss = self.gauss.reshape(img.shape[0],img.shape[1]).astype('uint8')
            noise = img + self.gauss
        else:
            gauss = np.random.normal(0,self.amount,img.size)
            gauss = gauss.reshape(img.shape[0],img.shape[1]).astype('uint8')
            noise = img + gauss
        
        return Image.fromarray(noise)
    
class Rotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        return TF.rotate(sample, self.angle)
    
class BrightnessContrast:
    def __init__(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        return TF.adjust_contrast(TF.adjust_brightness(sample, self.brightness),self.contrast)

