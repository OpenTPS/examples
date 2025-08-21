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
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from multiprocessing import shared_memory
from pathlib import Path

from opentps.core.examples.syntheticData import createSynthetic3DCT

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





