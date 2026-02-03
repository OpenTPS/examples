'''
Morphons Registration 
=========================
author: OpenTPS team

This example will present the basis of morphons registration with openTPS core.
running time: ~ 5 minutes
'''
#%% 
# Setting up the environment in google collab
#--------------------------------------------
import sys
if "google.colab" in sys.modules:
    from IPython import get_ipython
    get_ipython().system('git clone https://gitlab.com/openmcsquare/opentps.git')
    get_ipython().system('pip install ./opentps')
    get_ipython().system('pip install scipy==1.10.1')
    import opentps

#%%
#imports

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import os

#%%
#import the needed opentps.core packages

from opentps.core.data.images import CTImage
from opentps.core.processing.registration.registrationMorphons import RegistrationMorphons
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syntheticData import *

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
fixed_img = np.full((100, 100, 100), -1000)
fixed_img[25:75, 25:75, 25:75] = 0
fixed = CTImage(imageArray=fixed_img, name='fixed', origin=[0, 0, 0], spacing=[1, 1, 1])
moving_img = np.full((100, 100, 100), -1000)
moving_img[30:75, 35:75, 40:75] = 0
moving = CTImage(imageArray=moving_img, name='moving', origin=[0, 0, 0], spacing=[1, 1, 1])

#%%
# Perform the Morphons registration
#----------------------------------

start_time = time.time()
reg = RegistrationMorphons(fixed, moving, baseResolution=2.0, nbProcesses=1)
df = reg.compute()
processing_time = time.time() - start_time
print('Registration processing time was', processing_time, '\n')

#%%
# Compute the difference between the 2 images and check results of the registration
#----------------------------------------------------------------------------------

# COMPUTE IMAGE DIFFERENCE
df.resampleOn(moving, fillValue=-1024.)
diff_before = fixed.copy()
diff_before._imageArray = moving.imageArray - fixed.imageArray
diff_after = fixed.copy()
diff_after._imageArray = reg.deformed.imageArray - fixed.imageArray

# CHECK RESULTS
diff_before_sum = abs(diff_before.imageArray.sum())
diff_after_sum = abs(diff_after.imageArray.sum())
assert diff_before_sum-diff_after_sum > 0, f"Image difference is larger after registration"
assert abs(diff_after.imageArray[27,27,27])==0, f"Wrong target voxel difference after registration {diff_after.imageArray[27,27,27]} (expected 0)"

#%%
# Plot results
#----------------

fig, ax = plt.subplots(3, 3)
vmin = -1000
vmax = 1000
x_slice = round(fixed.imageArray.shape[0] / 2) - 1
y_slice = round(fixed.imageArray.shape[1] / 2) - 1
z_slice = round(fixed.imageArray.shape[2] / 2) - 1

# Plot X-Y field
u = df.velocity.imageArray[:, :, z_slice, 0]
v = df.velocity.imageArray[:, :, z_slice, 1]
u[0,0] = 1
ax[0, 0].imshow(reg.deformed.imageArray[:, :, z_slice].T[::1, ::1], cmap='gray', origin='upper', vmin=vmin, vmax=vmax)
ax[0, 0].quiver(u.T[::1, ::1], v.T[::1, ::1], alpha=0.2, color='red', angles='xy', scale_units='xy', scale=1)
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('y')
ax[0, 1].imshow(diff_before.imageArray[:, :, z_slice].T[::1, ::1], cmap='gray', origin='upper', vmin=2*vmin, vmax=2*vmax)
ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('y')
ax[0, 2].imshow(diff_after.imageArray[:, :, z_slice].T[::1, ::1], cmap='gray', origin='upper', vmin=2*vmin, vmax=2*vmax)
ax[0, 2].set_xlabel('x')
ax[0, 2].set_ylabel('y')

# Plot X-Z field
compX = df.velocity.imageArray[:, y_slice, :, 0]
compZ = df.velocity.imageArray[:, y_slice, :, 2]
compZ[0,0] = 1
ax[1, 0].imshow(reg.deformed.imageArray[:, y_slice, :].T[::1, ::1], cmap='gray', origin='upper', vmin=vmin, vmax=vmax)
ax[1, 0].quiver(compX.T[::1, ::1], compZ.T[::1, ::1], alpha=0.2, color='red', angles='xy', scale_units='xy', scale=1)
ax[1, 0].set_xlabel('x')
ax[1, 0].set_ylabel('z')
ax[1, 1].imshow(diff_before.imageArray[:, y_slice, :].T[::1, ::1], cmap='gray', origin='upper', vmin=2*vmin, vmax=2*vmax)
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel('z')
ax[1, 2].imshow(diff_after.imageArray[:, y_slice, :].T[::1, ::1], cmap='gray', origin='upper', vmin=2*vmin, vmax=2*vmax)
ax[1, 2].set_xlabel('x')
ax[1, 2].set_ylabel('z')

# Plot Y-Z field
compY = df.velocity.imageArray[x_slice, :, :, 1]
compZ = df.velocity.imageArray[x_slice, :, :, 2]
compZ[0,0] = 1
ax[2, 0].imshow(reg.deformed.imageArray[x_slice, :, :].T[::1, ::1], cmap='gray', origin='upper', vmin=vmin, vmax=vmax)
ax[2, 0].quiver(compY.T[::1, ::1], compZ.T[::1, ::1], alpha=0.2, color='red', angles='xy', scale_units='xy', scale=1)
ax[2, 0].set_xlabel('y')
ax[2, 0].set_ylabel('z')
ax[2, 1].imshow(diff_before.imageArray[x_slice, :, :].T[::1, ::1], cmap='gray', origin='upper', vmin=2*vmin, vmax=2*vmax)
ax[2, 1].set_xlabel('y')
ax[2, 1].set_ylabel('z')
ax[2, 2].imshow(diff_after.imageArray[x_slice, :, :].T[::1, ::1], cmap='gray', origin='upper', vmin=2*vmin, vmax=2*vmax)
ax[2, 2].set_xlabel('y')
ax[2, 2].set_ylabel('z')

ax[0, 0].title.set_text('Deformed image and deformation field')
ax[0, 1].title.set_text('Difference before registration')
ax[0, 2].title.set_text('Difference after registration')

plt.savefig(os.path.join(output_path, 'Example_Morphons.png'))

print('Morphons example completed')
plt.show()
