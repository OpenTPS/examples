'''
Midp
====
author: OpenTPS team

This example shows how to create a mid-position CT from a 4DCT and visualize it.

running time: ~ 6 minutes
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

from opentps.core.data.dynamicData._dynamic3DModel import Dynamic3DModel
from opentps.core.data.dynamicData._dynamic3DSequence import Dynamic3DSequence
from opentps.core.data.images import CTImage
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from syntheticData import *

logger = logging.getLogger(__name__)
#%%
# Output path
#------------

output_path = os.path.join(os.getcwd(), 'Output', 'ExampleMidP')
if not os.path.exists(output_path):
        os.makedirs(output_path)
logger.info('Files will be stored in {}'.format(output_path))

#%%
# Generate synthetic 4DCT And MidP
#---------------------------------

# GENERATE SYNTHETIC 4D INPUT SEQUENCE
CT4D = createSynthetic4DCT()

# GENERATE MIDP
Model4D = Dynamic3DModel()
startTime = time.time()
Model4D.computeMidPositionImage(CT4D, 0, tryGPU=True)
stopTime = time.time()
print('midP computed in ', np.round(stopTime - startTime, 2), 'seconds')

# GENERATE ADDITIONAL PHASES
im1 = Model4D.generate3DImage(0.5/4, amplitude=1, tryGPU=False)
im2 = Model4D.generate3DImage(2/4, amplitude=2.0, tryGPU=False)
im3 = Model4D.generate3DImage(2/4, amplitude=0.5, tryGPU=False)

#%%
# Display results
#----------------

fig, ax = plt.subplots(2, 4)
fig.tight_layout()
y_slice = 95
ax[0,0].imshow(CT4D.dyn3DImageList[0].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0,0].title.set_text('Phase 0')
ax[0,1].imshow(CT4D.dyn3DImageList[1].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0,1].title.set_text('Phase 1')
ax[0,2].imshow(CT4D.dyn3DImageList[2].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0,2].title.set_text('Phase 2')
ax[0,3].imshow(CT4D.dyn3DImageList[3].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0,3].title.set_text('Phase 3')
ax[1,0].imshow(Model4D.midp.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1,0].title.set_text('MidP image')
ax[1,1].imshow(im1.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1,1].title.set_text('phase 0.5 - amplitude 1')
ax[1,2].imshow(im2.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1,2].title.set_text('phase 2 - amplitude 2')
ax[1,3].imshow(im3.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1,3].title.set_text('phase 2 - amplitude 0.5')

plt.savefig(os.path.join(output_path, 'ExampleMidp.png')) 
print('MidP example completed')
plt.show()

