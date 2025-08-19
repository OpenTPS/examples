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
    ## !!! if you do not pass the image size and data type as argument to the child script, it must be known here
    arrayInScript2 = np.ndarray((170, 170, 100), dtype=int, buffer=existing_shm.buf)
    processImage(arrayInScript2)

    ## Delete array and close shared memory when finished
    del arrayInScript2
    existing_shm.close()