'''
Back and Forth
=========================
author: OpenTPS team

This example shows how two different python environements can be used together.
A parent script launches a child script which uses a different python
environement but shares the same image data. It is possible to pass commands back and forth between the two scripts.
The child script in this example simulates the use of an AI model. The first command passed to the child script
is to initialise the model (the neural network structure is created and its weights are loaded for example).
Then, later in the parent script, another command is passed to the child script
to use the AI model, multiple times in a row if necessary.

Important to note: the code executed in the child script must end with a print to send a response, else the script is
stuck waiting for the response

Key features:
- Use of multiple python envs in communicating scripts.
- Share RAM memory space between 2 scripts without the need to save and load data on/from hard drives.
- The possibility to initialise first, then later in the parent script, use the AI model.

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
from multiprocessing import shared_memory
from subprocess import Popen, PIPE

from opentps.core.examples.syntheticData import createSynthetic3DCT

#%%
# Set the child script environnement path and child scrip file path
childEnvPath = 'python.exe'  ## example: 'C:/Users/johnsmith/anaconda3/envs/myEnv/python.exe
childScriptPath = 'backAndForthChild.py'

#%%
# Create test image to share between scripts
ct = createSynthetic3DCT()
sliceToShow = 100

#%%
# Initialize shared memory and copy image array to this space
print(ct.imageArray.shape) ## These must be either passed to the child script as arguments or fixed and known
print(ct.imageArray.dtype) ## These must be either passed to the child script as arguments or fixed and known
print(ct.imageArray.nbytes) ## These must be either passed to the child script as arguments or fixed and known
shm = shared_memory.SharedMemory(create=True, size=ct.imageArray.nbytes, name='sharedArray')
sharedTestArray = np.ndarray(ct.imageArray.shape, dtype=ct.imageArray.dtype, buffer=shm.buf)
sharedTestArray[:] = ct.imageArray[:]

#%%
## Plot initial image
plt.figure()
plt.title("Before initialize")
plt.imshow(sharedTestArray[:, sliceToShow, :])
plt.show()

#%%
# Launch child process
process = Popen(childEnvPath + ' ' + childScriptPath, stdin=PIPE, stdout=PIPE, encoding='utf-8', text=True)#, universal_newlines=True, shell=True)

#%%
# Send the command 'init' to second process
process.stdin.write('init' + '\n')
process.stdin.flush()

#%%
# Get the response from the second script
response = process.stdout.readline().strip()
print(f'Back in script 1 after init command: Response: {response}')

print('Do something else in script 1')

#%%
# Plot image after init command
ct.imageArray[:] = sharedTestArray[:]
plt.figure()
plt.title("After initialize")
plt.imshow(ct.imageArray[:, sliceToShow, :])
plt.show()

for i in range(3):

    ## Send command 'processImage'
    process.stdin.write('processImage' + '\n')
    process.stdin.flush()

    ## Get the response from the second script
    response = process.stdout.readline().strip()
    print(f'Back in script 1 after command "processImage": Response: {response}')

    ## Plot image after process image command
    ct.imageArray[:] = sharedTestArray[:]
    plt.figure()
    plt.title("After process image")
    plt.imshow(ct.imageArray[:, sliceToShow, :])
    plt.show()

#%%
# Close the communication
process.stdin.close()
process.stdout.close()

#%%
# Close the shared memory
shm.close()
shm.unlink()

print('End of script 1')
