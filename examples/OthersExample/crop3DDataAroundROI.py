'''
Crop 3D Data Around ROI
=========================
author: OpenTPS team

This example shows how to: 
- Read a serialized patient with a Dynamic3DSequence, a Dynamic3DModel and an RTStruct
!! The data is not given in the test data folder of the project !!
- Select an ROI from the RTStruct object
- Get the ROI as an ROIMask
- Get the box around the ROI in scanner coordinates
- Crop the dynamic sequence and the dynamic model around the box

running time: ~ 10 minutes
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
from opentps.core.processing.imageProcessing.resampler3D import crop3DDataAroundBox
from opentps.core.processing.segmentation.segmentation3D import getBoxAroundROI
from opentps.core.io.serializedObjectIO import loadDataStructure

#%%
# Load the serialized patient data
dataPath = '/data/Patient0BaseAndMod.p'
patient = loadDataStructure(dataPath)[0]

dynSeq = patient.getPatientDataOfType("Dynamic3DSequence")[0]
dynMod = patient.getPatientDataOfType("Dynamic3DModel")[0]
rtStruct = patient.getPatientDataOfType("RTStruct")[0]

#%%
# get the ROI and mask on which we want to apply the motion signal
print('Available ROIs')
rtStruct.print_ROINames()
bodyContour = rtStruct.getContourByName('body')
ROIMask = bodyContour.getBinaryMask(origin=dynMod.midp.origin, gridSize=dynMod.midp.gridSize, spacing=dynMod.midp.spacing)

#%%
# get the box around the ROI
box = getBoxAroundROI(ROIMask)
marginInMM = [10, 10, 10]
#%%
# crop the dynamic sequence and the dynamic model around the box
crop3DDataAroundBox(dynSeq, box, marginInMM=marginInMM)
print('-'*50)
crop3DDataAroundBox(dynMod, box, marginInMM=marginInMM) 
