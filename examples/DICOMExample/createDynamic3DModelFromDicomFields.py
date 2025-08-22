'''
Create Dynamic 3D Model from DICOM Fields
=========================================
author: OpenTPS team

This example shows how to read a DICOM CT and deformation fields, create a dynamic 3D model with the mid-position CT and the deformation fields, and print the model information.

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
import os
import sys

#%%
#import the needed opentps.core packages
from opentps.core.io.dataLoader import readData
from opentps.core.data.images._deformation3D import Deformation3D
from opentps.core.data.dynamicData._dynamic3DModel import Dynamic3DModel

#%%
# Load DICOM CT
inputPaths = 'Path_to_your_CT_data/'  # replace with the path to your CT data folder
dataList = readData(inputPaths, maxDepth=0)
midP = dataList[0]
print(type(midP))

#%%
# Load DICOM Deformation Fields
inputPaths = 'Path_to_your_deformation_fields/'  # replace with the path to your deformation fields folder
defList = readData(inputPaths, maxDepth=0)

#%%
# Transform VectorField3D to deformation3D
deformationList = []
for df in defList:
    df2 = Deformation3D()
    df2.initFromVelocityField(df)
    deformationList.append(df2)
del defList
print(deformationList)

patient_name = 'OpenTPS_Patient'

#%%
# Create Dynamic 3D Model
model3D = Dynamic3DModel(name=patient_name, midp=midP, deformationList=deformationList)
print(model3D)