import random
import warnings
import numbers

import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
import cv2


class RandomCrop:

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        h, w = next(iter(sample.values())).size
        try:
            i = random.randrange(0, h-self.size[0])
        except ValueError: # empty range
            warnings.warn("height too small {} cropping. Adjusting parameters".format(h))
            i = h
        try:
            j = random.randrange(0, w-self.size[1])
        except ValueError: # empty range
            warnings.warn("width too small {} cropping. Adjusting parameters".format(w))
            j = w
        return {k: TF.crop(v, i, j, *self.size) for k, v in sample.items()}

class ToTensor:
    def __call__(self, sample):
        #return {k: TF.to_tensor(v) for k, v in sample.items()}
        return TF.to_tensor(sample['in']), TF.to_tensor(sample['out'])
        
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        p (float): Probability to perform cutout.
    """
    def __init__(self, n_holes, length, p, labels):
        self.n_holes = n_holes
        self.length = length
        self.p = p
        self.labels = labels

    def __call__(self, sample):
        wl, td, bc, sl = sample['wl'], sample['td'], sample['bc'], sample['sl']
        
        w, h = wl.size

        mask = np.ones((h,w))

        if np.random.rand(1)[0] <= self.p:
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            if 'wl' in self.labels: wl = Image.fromarray(wl * mask)
            if 'td' in self.labels: td = Image.fromarray(td * mask)
            if 'bc' in self.labels: bc = Image.fromarray((bc * mask).astype('uint8'))
            if 'sl' in self.labels: sl = Image.fromarray((sl * mask).astype('uint8'))

        return {'wl': wl, 'td': td, 'bc': bc, 'sl': sl}
    
class AdditiveNoise(object):
    def __init__(self, p, scale, labels):
        self.p = p
        self.scale = scale
        self.labels = labels

    def __call__(self, sample):
        
        for label in self.labels:
            tmp = sample[label]
            w, h = tmp.size
            tmp = np.array(tmp)
            vmax = np.max(tmp)
            if self.scale == 1:
                tmp[np.random.rand(h,w)>self.p] = vmax
            else:
                tmp[cv2.resize(np.uint8(np.random.rand(int(h/self.scale),int(w/self.scale))>self.p),(w,h))>0] = vmax
            
            if label == 'bc':
                sample[label] = Image.fromarray(tmp.astype('uint8'))
            else:
                sample[label] = Image.fromarray(tmp)

        return sample
    
class Stack(object):
    def __init__(self, ins, outs):
        self.ins = ins
        self.outs = outs

    def __call__(self, sample):
            
        inputs = [np.array(sample[v]) for v in self.ins]
        input = np.stack(inputs, axis=-1)
        
        if len(self.outs) > 1:
            outputs = [np.array(sample[v]) for v in self.outs]
            output = np.stack(outputs, axis=-1)
        else:
            output = sample[self.outs[0]]

        return {'in': input, 'out': output}
