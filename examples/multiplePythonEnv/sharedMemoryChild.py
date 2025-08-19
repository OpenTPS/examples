'''
Shared Memory child 
===================
author: OpenTPS team

This script is the child script that will be launched by the parent script.
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
import numpy as np
from multiprocessing import shared_memory

#%%
def processImage(img):
    ## This is the function that will be applied to the shared image
    targetPos = [120, 100, 45]
    img[targetPos[0] - 10:targetPos[0] + 10, targetPos[1] - 10:targetPos[1] + 10, targetPos[2] - 10:targetPos[2] + 10] = 0

#%%
if __name__ == "__main__":

    print("Start script 2")
    existing_shm = shared_memory.SharedMemory(name='sharedArray')
    try:
        # Must match parent array shape and dtype
        arrayInScript2 = np.ndarray((170, 170, 100), dtype=np.int64, buffer=existing_shm.buf)
        processImage(arrayInScript2)
    finally:
        del arrayInScript2
        existing_shm.close()
