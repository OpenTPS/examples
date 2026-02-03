'''
Generate random samples from a model
====================================
author: OpenTPS team

This example demonstrates how to generate random samples from a model using OpenTPS.

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
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import opentps

#%%
# import the needed opentps.core packages
from opentps.core.io.serializedObjectIO import loadDataStructure
from opentps.core.processing.deformableDataAugmentationToolBox.generateRandomSamplesFromModel import generateRandomImagesFromModel, generateRandomDeformationsFromModel

#%%
#Data path
testDataPath = os.path.join(Path(os.getcwd()).parent.absolute(), 'testData/')

#%%
# read a serialized dynamic sequence
dataPath = testDataPath + "veryLightDynMod.p"
dynMod = loadDataStructure(dataPath)[0]

tryGPU = True
numberOfSamples = 50

imageList = []

startTime = time.time()

defList = generateRandomDeformationsFromModel(dynMod, numberOfSamples=numberOfSamples, ampDistribution='gaussian')
for deformation in defList:
    imageList.append(deformation.deformImage(dynMod.midp, fillValue='closest', tryGPU=tryGPU))

print(len(imageList))
print('first test done in ', np.round(time.time() - startTime, 2))

plt.figure()
plt.imshow(imageList[0].imageArray[:, 50, :])
plt.show()

#%%
startTime = time.time()
imageList = generateRandomImagesFromModel(dynMod, numberOfSamples=numberOfSamples, ampDistribution='gaussian', tryGPU=tryGPU)
print('second test done in ', np.round(time.time() - startTime, 2))

plt.figure()
plt.imshow(imageList[0].imageArray[:, 50, :])
plt.show()