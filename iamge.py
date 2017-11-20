# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:33:22 2017

@author: yang
"""

from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

imgpath = "new-traffic-signs/11.jpg"
image = mpimg.imread(imgpath)

#Convert to single channel Y
y_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

plt.figure()
plt.subplot(2,2,1)
plt.imshow(image)
plt.subplot(2,2,2)
plt.imshow(y_image,cmap='gray')

nom_image = (y_image / 255.).astype(np.float32)
sharp_image = exposure.equalize_adapthist(nom_image)
sharp_image = (sharp_image*255)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(nom_image,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(sharp_image,cmap='gray')