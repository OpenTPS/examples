'''
Create Model with ROIs from DICOM Images
========================================
author: OpenTPS team

This example shows how to: 
- read dicom data from a 4DCT folder
- create a dynamic 3D sequence with the 4DCT data
- read an rtStruct dicom file
- create a dynamic 3D model and compute the midP image with the dynamic 3D sequence
- create a patient, give him the model and rtStruct and save it as serialized data  
!!! does not work with public data for now since there is no struct in the public data !!!

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
import time
import numpy as np

#%%
#import the needed opentps.core packages
from pydicom.uid import generate_uid
from opentps.core.io.dataLoader import readData
from opentps.core.data.dynamicData._dynamic3DSequence import Dynamic3DSequence
from opentps.core.io.serializedObjectIO import saveSerializedObjects
from opentps.core.data.dynamicData._dynamic3DModel import Dynamic3DModel
from opentps.core.data._patient import Patient

#%%
# chose the patient folder, which will be used as the patient name
patientName = 'Patient_0'

#%%
# chose the 4DCT data folder
data4DPath = 'Path_to_your_4DCT_data/'  # replace with the path to your 4DCT data folder
# chose the dicom rtStruct file
dataStructPath = 'Path_to_your_rtStruct_data/'  # replace with the path to your rtStruct data folder
# chose a path to save the results
savingPath = 'Path_to_your_saving_path/'  # replace with the path where you want to save the results
#%%
# load the 4DCT data
data4DList = readData(data4DPath)
print(len(data4DList), 'images found in the folder')
print('Image type =', type(data4DList[0]))
print('Image 0 shape =', data4DList[0].gridSize)

# create a Dynamic3DSequence and change its name
dynSeq = Dynamic3DSequence(dyn3DImageList=data4DList)
dynSeq.name = '4DCT'

#%%
# load the rtStruct data and print its content
structData = readData(dataStructPath)[0]
print('Available ROIs')
structData.print_ROINames()

#%%
# create Dynamic3DModel
model3D = Dynamic3DModel()

# change its name
model3D.name = 'MidP'

# give it an seriesInstanceUID
model3D.seriesInstanceUID = generate_uid()

# give it an seriesInstanceUID
model3D.seriesInstanceUID = generate_uid()

# generate the midP image and deformation fields from the dynamic 3D sequence
startTime = time.time()
model3D.computeMidPositionImage(dynSeq, tryGPU=True)
stopTime = time.time()

print(model3D.midp.name)
print('MidP computed in ', np.round(stopTime-startTime))

#%%
# Create a patient and give it the patient name
patient = Patient()
patient.name = patientName

#%%
# Add the model and rtStruct to the patient
patient.appendPatientData(model3D)
patient.appendPatientData(structData)

#%%
## Save it as a serialized object
saveSerializedObjects(patient, savingPath)
