import skimage as ski
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from skimage.draw import disk, polygon, ellipse
import numpy as np
import torch
import random


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def tv(img):      
    tv_h = ((img[:,:,1:,:] - img[:,:,:-1,:])).abs().sum()
    tv_w = ((img[:,:,:,1:] - img[:,:,:,:-1])).abs().sum()    
    return (tv_h + tv_w)/(img.shape[-1] * img.shape[-2])

def get_phantom(dim):
    phantom = ski.img_as_float(ski.data.shepp_logan_phantom())
    return ski.transform.resize(phantom, (dim, dim))

class random_weighted_norm:
    def __init__(self, im_size = 64, pos_var = 0.1, w_max = 2, r = None):
        x,y= torch.meshgrid(*[torch.linspace(-1,1, im_size) for _ in [0,1]], indexing='ij')
        self.xy = torch.stack([x,y])
        self.pos_var = pos_var
        self.w_max = w_max
        self.r = (0.3,0.6) if r is None else r
          
    def __call__(self,p=1):
        w = np.random.uniform(1., self.w_max, size=(2,))
        r = random.uniform(*self.r)
        m = np.random.normal(0, self.pos_var, size=(2,1,1))
        return self.weighted_norm(m, w, p, r)
    
    def weighted_norm(self, m, w, p, r):
        return 1. * (torch.linalg.norm((self.xy - m) * w[:,None,None], dim=0, ord=p) < r)
    


class shapes:
    def __init__(self, img_size=100, p=0., noise_lvl=0.):
        self.img_size = [img_size, img_size]
        self.p = p
        self.noise_lvl = noise_lvl
        self.shape_names = ['rectangle', 'circle', 'triangle', 'ellipse']
        
    def get_shape(self, name='rectangle'):
        if name == 'rectangle':
            I =  self.rectangle()
        elif name == 'circle':
            I = self.circle()
        elif name == 'triangle':
            I = self.triangle()
        elif name == 'ellipse':
            I = self.ellipse()
        elif name == 'random':
            i = np.random.randint(0, len(self.shape_names)-1)
            shape_name = self.shape_names[i]
            I = self.get_shape(name=shape_name)
        else:
            raise RuntimeError('Unknown shape: ' + str(name))
            
            
        return I

    def rectangle(self,):
        S = np.zeros(self.img_size)
        x_start, y_start = np.random.randint(self.img_size[1]//4,self.img_size[1]//3 , size=(2,))
        x_end, y_end = np.random.randint(self.img_size[1]//2, 3*self.img_size[1]//4, size=(2,))
        S[x_start:x_end, y_start:y_end] = 1
        return S
        
    def circle(self,):
        C = np.zeros(self.img_size)
        radius = np.random.randint(self.img_size[1]//5, self.img_size[1]//4)
        row = self.img_size[1]//2# + np.random.randint(-self.img_size[1]//6, self.img_size[1]//6)
        col = self.img_size[1]//2# + np.random.randint(-self.img_size[1]//6, self.img_size[1]//6)
        # modern scikit uses a tuple for center
        rr, cc = disk((row, col), radius)
        C[rr, cc] = 1.
        return C
    
    def triangle(self,):
        T = np.zeros(self.img_size)
        
        poly = np.random.randint(self.img_size[0]//4, 3 * self.img_size[0]//4, (3,2))
        rr, cc = polygon(poly[:, 0], poly[:, 1], T.shape)
        T[rr, cc] = 1
        return T
    
    def ellipse(self,):
        E = np.zeros(self.img_size)
        r_radius = np.random.randint(self.img_size[1]//5, self.img_size[1]//4)
        c_radius = np.random.randint(self.img_size[1]//5, self.img_size[1]//4)

        row = self.img_size[1]//2# + np.random.randint(-self.img_size[1]//6, self.img_size[1]//6)
        col = self.img_size[1]//2# + np.random.randint(-self.img_size[1]//6, self.img_size[1]//6)
        
        rot = np.random.uniform(-np.pi, np.pi)
        
        rr, cc = ellipse(row, col, r_radius, c_radius, shape=None, rotation=rot)
        E[rr, cc] = 1.
        return E