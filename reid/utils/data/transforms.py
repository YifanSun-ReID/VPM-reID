from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import pdb
import numpy as np

class RandomVerticalCrop(object):
    def __init__(self, height,width):
        self.height = height
        self.width = width
    def __call__(self, img):
        img = img.resize((self.width, self.height),Image.BILINEAR)
        rp = np.random.random_integers(3,6)
        img = img.crop((0,0,self.width,self.height/6.*rp))
        img = img.resize((self.width, self.height),Image.BILINEAR)
        return [img, rp]

class RandomVerticalCropCont(object):
    def __init__(self, height,width):
        self.height = height
        self.width = width
    def __call__(self, img):
        w, h = img.size
        ratio = min(1,np.random.uniform(0.5, 1.08333))
        ratio = float(ratio)#
        jitter = np.random.uniform(0.9,1.11111)
        apply_ratio = min(1.0, ratio*jitter) 
        apply_ratio = ratio 
        start_ratio = 0
        if np.random.uniform(0,1) > 1.0:   # the probability of Top-started crop is 1.0 
            start_ratio = 1-apply_ratio
        start_h = int(start_ratio*h)
        img = img.crop((0, start_h, w, np.round(h*apply_ratio)))
        img = img.resize((self.width, self.height),Image.BILINEAR)
        return [img, (apply_ratio, start_ratio)]

class ContVerticalCropDiscret(object):
    def __init__(self, height,width, ratio):
        self.height = height
        self.width = width
        self.ratio = ratio
    def __call__(self, img):
        w, h  = img.size
        img = img.crop((0,0, w, np.round(h*self.ratio)))
        img = img.resize((self.width, self.height),Image.BILINEAR)
        return img
    
class ContVerticalCrop(object):
    def __init__(self, height,width,ratio):
        self.height = height
        self.width = width
        self.ratio = ratio
    def __call__(self, img):
        w, h = img.size
        rp = np.random.uniform(self.ratio, 1)
        img = img.crop((0, 0, w, np.round(h*rp)))
        img = img.resize((self.width, self.height),Image.BILINEAR)
        return img

class ListToTensor(object):
    def __init__(self):
        self.totensor = ToTensor()
    def __call__(self,img_rp):
        tensor =  self.totensor(img_rp[0])
        return [tensor, img_rp[1]]

class ListNormalize(object):
    def __init__(self, mean, std):
        self.normalizer = Normalize(mean,std)
    def __call__(self,tensor_rp):
        return [self.normalizer(tensor_rp[0]), tensor_rp[1]]


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)
