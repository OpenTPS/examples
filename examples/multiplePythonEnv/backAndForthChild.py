'''
Back and Forth Child
=========================
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
import sys
from multiprocessing import shared_memory
import matplotlib.pyplot as plt

#%%
class AIModel():

    def __init__(self, image=np.zeros((30, 40, 50)), name='test'):

        self.img = image
        self.testVariable = 0

    def processImage(self):

        self.testVariable += 1
        targetPos = [120, 100, 45]
        self.img[targetPos[0] - 10:targetPos[0] + 10, targetPos[1] - 10:targetPos[1] + 10,
        targetPos[2] - 10:targetPos[2] + 10] += 500


#%%
if __name__ == "__main__":

    for line in sys.stdin:
        command = line.strip()
        # print(command)
        # Process the command
        # You can perform specific tasks based on the received command
        # For example:
        if command == 'init':
            existing_shm = shared_memory.SharedMemory(name='sharedArray')
            img = np.ndarray((170, 170, 100), dtype=int, buffer=existing_shm.buf)
            aimodel = AIModel(image=img)
            ## !!! Without this print the response is never sent and the script is blocked
            print('Initialize AI Model. testVariable =', aimodel.testVariable)
        else:
            method = getattr(aimodel, command)
            method()
            ## !!! Without this print the response is never sent and the script is blocked
            print(f'Executing {command} of AI Model. testVariable =', aimodel.testVariable)

        # Flush the output to ensure the parent script receives it
        sys.stdout.flush()


    # print("Start script 2")
    # existing_shm = shared_memory.SharedMemory(name='sharedArray')
    # arrayInScript2 = np.ndarray((30, 40, 50), dtype=float, buffer=existing_shm.buf)
    # processedArray = script2(arrayInScript2)
    # del arrayInScript2
    # existing_shm.close()