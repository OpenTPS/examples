'''
Create 3D Sequence from DICOM
=============================
author: OpenTPS team

This example shows how to read data from a 4DCT folder, create a dynamic 3D sequence with the 4DCT data, and save this sequence in serialized format on the drive.
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
import os
from pathlib import Path
import sys

#%%
#import the needed opentps.core packages

from opentps.core.io.dataLoader import readData
from opentps.core.data.dynamicData._dynamic3DSequence import Dynamic3DSequence
from opentps.core.io.serializedObjectIO import saveSerializedObjects

#%%
# Get the current working directory, its parent, then add the testData folder at the end of it
testDataPath = os.path.join(Path(os.getcwd()).parent.absolute(), 'opentps/testData/')

#%%
## read a serialized dynamic sequence
dataPath = testDataPath + "4DCTDicomLight"

print('Datas present in ' + dataPath + 'are loaded.')
dataList = readData(dataPath)
print(len(dataList), 'images found in the folder')
print('Image type =', type(dataList[0]))

#%%
# create a Dynamic3DSequence and change its name
dynseq = Dynamic3DSequence(dyn3DImageList=dataList)
print('Type of the created object =', type(dynseq))
print('Sequence name =', dynseq.name)
dynseq.name = 'Light4DCT'
print('Sequence name = ', dynseq.name)
print('Sequence lenght =', len(dynseq.dyn3DImageList))

#%%
# save it as a serialized object
savingPath = testDataPath + 'lightDynSeq'
saveSerializedObjects(dynseq, savingPath)
