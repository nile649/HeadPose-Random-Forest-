import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import pdb
class GaborData:
    def __init__(self):
        self.__gaborFilterBank16__()
    def __patches__(self,x):
        rng = np.random.RandomState(0)
        patch_size = (48, 48)
        data = []
        data.append(x[0:48,0:48])
        data.append(x[48:96,0:48])
        data.append(x[0:48,48:96])
        data.append(x[48-48//2:48+48//2,48-48//2:48+48//2])
        data.append(x[48:96,48:96])
        
        return data
    def __compute_feats__(self,image):
        feats = np.zeros((len(self.filter), 2), dtype=np.double)
        for k, kernel in enumerate(self.filter):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats

    def __gaborFilterBank16__(self):
        self.filter = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    self.filter.append(kernel)
        
    
    def getGabor(self,x):
        features = np.zeros((1,10))
        data = self.__patches__(x)
        for i,d in enumerate(data):
            v = sum(self.__compute_feats__(d))
#             v = v / np.sqrt(np.sum(v**2))
            features[0][i*2] = v[0]
            features[0][i*2+1] = v[1]
        return features
