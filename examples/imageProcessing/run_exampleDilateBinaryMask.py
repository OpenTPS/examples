'''
Dilate Binary Mask 
=========================
author: OpenTPS team


'''
#%% 
# Setting up the environment in google collab
#--------------
# First you need to change the type of execution in the bottom left from processor to GPU. Then you can run the example.
import sys
if "google.colab" in sys.modules:
    from IPython import get_ipython
    get_ipython().system('git clone https://gitlab.com/openmcsquare/opentps.git')
    get_ipython().system('pip install ./opentps')
    get_ipython().system('pip install scipy==1.10.1')
    get_ipython().system('pip install cupy-cuda12x')
    import opentps

#%%
#imports
import numpy as np
import matplotlib.pyplot as plt

#%%
#import the needed opentps.core packages

from opentps.core.data.images._roiMask import ROIMask
from opentps.core.processing.imageProcessing.roiMasksProcessing import buildStructElem, dilateMaskScipy
import os
import logging

logger = logging.getLogger(__name__)

#%%
# Output path
output_path = os.path.join(os.getcwd(), 'Output', 'ExampleDilateBinaryMask')
if not os.path.exists(output_path):
        os.makedirs(output_path)
logger.info('Files will be stored in {}'.format(output_path))

#%%
# Create a synthetic 3D ROI mask
roi = ROIMask(name='TV')
roi.color = (255, 0, 0)# red
data = np.zeros((100, 100, 100)).astype(bool)
data[50:60, 50:60, 50:60] = True
roi.imageArray = data
roi.spacing = np.array([1, 1, 2])

radius = np.array([4, 4, 6])
struct = buildStructElem(radius / np.array(roi.spacing))

roi_scipy = roi.copy()
dilateMaskScipy(roi_scipy, radius=radius)  # scipy
print(radius, 'before roi_sitk')
roi_sitk = roi.copy()
roi_sitk.dilateMask(radius=radius)

#%%
# Visualize the results

plt.figure()
plt.subplot(2, 4, 1)
plt.imshow(roi.imageArray[55, :, :], cmap='gray')
plt.title("Original")

plt.subplot(2, 4, 2)
plt.imshow(roi_scipy.imageArray[55, :, :], cmap='gray')
plt.title("Scipy")

plt.subplot(2, 4, 3)
plt.imshow(roi_sitk.imageArray[55, :, :], cmap='gray')
plt.title("SITK")

plt.subplot(2, 4, 5)
plt.imshow(roi_scipy.imageArray[55, :, :] ^ roi_sitk.imageArray[55, :, :], cmap='gray')
plt.title("diff Scipy-SITK")

plt.subplot(2, 4, 6)
plt.imshow(roi_scipy.imageArray[55, :, :] ^ roi.imageArray[55, :, :], cmap='gray')
plt.title("diff Scipy-ori")

plt.subplot(2, 4, 7)
plt.imshow(roi_sitk.imageArray[55, :, :] ^ roi.imageArray[55, :, :], cmap='gray')
plt.title("diff SITK-ori")

plt.savefig(os.path.join(output_path, 'ExampleDilateBinary.png')) 
plt.show()
