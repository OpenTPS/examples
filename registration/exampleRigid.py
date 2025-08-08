'''
Rigid Registration 
=========================
author: OpenTPS team

This example will present the basis of rigid registration with openTPS core.
'''
# %% [skip]
# installing opentps in colab
import sys
if "google.colab" in sys.modules:
    from IPython import get_ipython
    get_ipython().system('git clone https://gitlab.com/openmcsquare/opentps.git')
    get_ipython().system('pip install ./opentps')
    import opentps

#%%
#imports
import copy

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import os

#%%
#import the needed opentps.core packages

from opentps.core.processing.registration.registrationRigid import RegistrationRigid
from opentps.core.examples.syntheticData import *
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
from opentps.core.processing.imageProcessing.imageTransform3D import rotateData, translateData

logger = logging.getLogger(__name__)

#%%
#Output path
#-----------
output_path = 'Output'
if not os.path.exists(output_path):
    os.makedirs(output_path)
logger.info('Files will be stored in {}'.format(output_path))

#%%
# Generate synthetic images
#--------------------------

fixed = createSynthetic3DCT()
moving = copy.copy(fixed)

translation = np.array([15, 0, 10])
rotation = np.array([0, 5, 2])

translateData(moving, translation, outputBox='same')
rotateData(moving, rotation, outputBox='same')

#%%
# Perform the Rigid registration
#-------------------------------

start_time = time.time()
reg = RegistrationRigid(fixed, moving)
transform = reg.compute()

processing_time = time.time() - start_time
print('Registration processing time was', processing_time, '\n')
print('Translation', transform.getTranslation())
print('Rotation in deg', transform.getRotationAngles(inDegrees=True), '\n')

## Two ways of getting the deformed moving image
deformedImage = reg.deformed
# deformedImage = transform.deformImage(moving)

## Resample it to the same grid as the fixed image
resampledOnFixedGrid = resampleImage3DOnImage3D(deformedImage, fixedImage=fixed, fillValue=-1000)

#%%
# Compute the difference between the 2 images and check results of the registration
#----------------------------------------------------------------------------------

# COMPUTE IMAGE DIFFERENCE
diff_before = fixed.copy()
diff_before._imageArray = fixed.imageArray - moving.imageArray
diff_after = fixed.copy()
diff_after._imageArray = fixed.imageArray - resampledOnFixedGrid.imageArray

# CHECK RESULTS
diff_before_sum = abs(diff_before.imageArray).sum()
diff_after_sum = abs(diff_after.imageArray).sum()
assert diff_before_sum - diff_after_sum > 0, f"Image difference is larger after registration"

#%%
#Plot results
#-------------
y_slice = 95
fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(fixed.imageArray[:, y_slice, :])
ax[0, 0].set_title('Fixed')
ax[0, 0].set_xlabel('Origin: '+f'{fixed.origin[0]}'+','+f'{fixed.origin[1]}'+','+f'{fixed.origin[2]}')
ax[0, 1].imshow(moving.imageArray[:, y_slice, :])
ax[0, 1].set_title('Moving')
ax[0, 1].set_xlabel('Origin: ' + f'{moving.origin[0]}' + ',' + f'{moving.origin[1]}' + ',' + f'{moving.origin[2]}')
diffBef = ax[0, 2].imshow(diff_before.imageArray[:, y_slice, :], vmin=-2000, vmax=2000)
ax[0, 2].set_title('Diff before')
fig.colorbar(diffBef, ax=ax[0, 2])
ax[1, 0].imshow(deformedImage.imageArray[:, y_slice, :])
ax[1, 0].set_title('DeformedMoving')
ax[1, 0].set_xlabel('Origin: ' + f'{deformedImage.origin[0]:.1f}' + ',' + f'{deformedImage.origin[1]:.1f}' + ',' + f'{deformedImage.origin[2]:.1f}')
ax[1, 1].imshow(resampledOnFixedGrid.imageArray[:, y_slice, :])
ax[1, 1].set_title('resampledOnFixedGrid')
ax[1, 1].set_xlabel('Origin: ' + f'{resampledOnFixedGrid.origin[0]:.1f}' + ',' + f'{resampledOnFixedGrid.origin[1]:.1f}' + ',' + f'{resampledOnFixedGrid.origin[2]:.1f}')
diffAft = ax[1, 2].imshow(diff_after.imageArray[:, y_slice, :], vmin=-2000, vmax=2000)
ax[1, 2].set_title('Diff after')
fig.colorbar(diffAft, ax=ax[1, 2])
plt.savefig(os.path.join(output_path, 'Example_Registration.png'))

print('Rigid registration example completed')
plt.show()