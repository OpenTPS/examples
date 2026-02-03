'''
Shared Memory 
=============
author: OpenTPS team

This example shows how two different python environnements can be used together.
A parent script launches a child script which uses a different python
environement but shares the same image data.

Key features:
- The parent script launches a child script which use a different python environnement
- Share RAM memory space between 2 scripts without the need to save and load data on/from hard drives.

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

    import opentps

#%%
#imports
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from multiprocessing import shared_memory
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))



#%%
#import the needed opentps.core packages
from opentps.core.data.images import CTImage
from opentps.core.data.images import ROIMask

#%%
# Synthetic 4DCT generation function
#----------------------------------------

def createSynthetic3DCT(diaphragmPos = 20, targetPos = [50, 100, 35], spacing=[1, 1, 2], returnTumorMask = False):
    # GENERATE SYNTHETIC CT IMAGE
    # background
    im = np.full((170, 170, 100), -1000)
    im[20:150, 70:130, :] = 0
    # left lung
    im[30:70, 80:120, diaphragmPos:] = -800
    # right lung
    im[100:140, 80:120, diaphragmPos:] = -800
    # target
    im[targetPos[0]-5:targetPos[0]+5, targetPos[1]-5:targetPos[1]+5, targetPos[2]-5:targetPos[2]+5] = 0
    # vertebral column
    im[80:90, 95:105, :] = 800
    # rib
    im[22:26, 90:110, 46:50] = 800
    # couch
    im[:, 130:135, :] = 100
    ct = CTImage(imageArray=im, name='fixed', origin=[0, 0, 0], spacing=spacing)

    if returnTumorMask:
        mask = np.full((170, 170, 100), 0)
        mask[targetPos[0]-5:targetPos[0]+5, targetPos[1]-5:targetPos[1]+5, targetPos[2]-5:targetPos[2]+5] = 1
        roi = ROIMask(imageArray=mask, origin=[0, 0, 0], spacing=spacing)

        return ct, roi

    else:
        return ct
#%%
# set the child script environnement path and child scrip file path
script2EnvPath = sys.executable  # absolute path to current python
script2Path = str(Path.cwd() / "sharedMemoryChild.py")

#%%
# create test image to share between scripts
ct = createSynthetic3DCT()
sliceToShow = 100

#%%
# initialize shared memory and copy image array to this space
print(ct.imageArray.shape) ## these must be either passed to the child script as arguments or fixed and known
print(ct.imageArray.dtype)
print(ct.imageArray.nbytes)
shm = shared_memory.SharedMemory(create=True, size=ct.imageArray.nbytes, name='sharedArray')
sharedTestArray = np.ndarray(ct.imageArray.shape, dtype=ct.imageArray.dtype, buffer=shm.buf)
sharedTestArray[:] = ct.imageArray[:]

#%%
# plot image before child script call
plt.figure()
plt.title("Before subprocess")
plt.imshow(ct.imageArray[:, sliceToShow, :])
plt.show()

#%%
# Call to child script
subprocess.call([script2EnvPath, script2Path])

#%%
# Copy shared memory space to test image array
ct.imageArray[:] = sharedTestArray[:]

#%%
# Plot image after child script call
plt.figure()
plt.title("After subprocess")
plt.imshow(ct.imageArray[:, sliceToShow, :])
plt.show()

#%%
# Close the shared memory
shm.close()
try:
    shm.unlink()
except FileNotFoundError:
    print("Shared memory already unlinked, skipping.")





