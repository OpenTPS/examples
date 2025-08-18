'''
Generate DRR and GTV Masks
==========================
author: OpenTPS team

This example shows how to:
- read model + ROI data from a serialized file
- create a breathing signal using the motion amplitude present in the model
- chose an ROI to apply the breathing signal to its center of mass
-

!!! does not work with public data for now since there is no struct in the public data !!!
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
from scipy.ndimage import zoom
import math
import time
import concurrent
from itertools import repeat
import os
import sys
import numpy as np

#%% 
# import the needed opentps.core packages

from opentps.core.processing.imageProcessing.resampler3D import crop3DDataAroundBox
from opentps.core.io.serializedObjectIO import saveSerializedObjects, loadDataStructure
from opentps.core.data.dynamicData._breathingSignals import SyntheticBreathingSignal
from opentps.core.processing.deformableDataAugmentationToolBox.generateDynamicSequencesFromModel import generateDeformationListFromBreathingSignalsAndModel
from opentps.core.processing.imageSimulation.DRRToolBox import forwardProjection
from opentps.core.processing.imageProcessing.image2DManip import getBinaryMaskFromROIDRR, get2DMaskCenterOfMass
from opentps.core.processing.imageProcessing.crop2D import getBoxAroundROI
from opentps.core.processing.imageProcessing.imageTransform3D import getVoxelIndexFromPosition
from opentps.core.processing.deformableDataAugmentationToolBox.modelManipFunctions import getAverageModelValuesAroundPosition

#%%
#paths selection

patientFolder = 'Patient_0'
patientFolderComplement = ''
organ = 'liver'
basePath = 'D:/ImageData/'

dataSetFolder = '/test/'
dataSetDataFolder = 'data/'

dataPath = basePath + organ + '/' + patientFolder + patientFolderComplement + '/dynModAndROIs.p'
savingPath = basePath + organ + '/' + patientFolder + patientFolderComplement + dataSetFolder

if not os.path.exists(savingPath):
    os.umask(0)
    os.makedirs(savingPath)   # Create a new directory because it does not exist
    os.makedirs(savingPath + dataSetDataFolder)  # Create a new directory because it does not exist
    print("New directory created to save the data: ", savingPath)

#%%
# parameters selection
sequenceDurationInSecs = 10
samplingFrequency = 4
subSequenceSize = 50
outputSize = [64, 64]
bodyContourToUse = 'body'
otherContourToUse = 'MidP CT GTV'
marginInMM = [50, 0, 100]

#%%
# breathing signal parameters
amplitude = 'model'
variationAmplitude = 2
breathingPeriod = 4
variationFrequency = 0.1
shift = 2
meanNoise = 0
varianceNoise = 0.5
samplingPeriod = 1 / samplingFrequency
simulationTime = sequenceDurationInSecs
meanEvent = 2 / 30
# use Z - 0 for Coronal and Z - 90 for sagittal
projAngle = 0
projAxis = 'Z'

multiprocessing = False
maxMultiProcUse = 4
tryGPU = True

#%%
# This function is specific to this example and used to :
#- deform a CTImage and an ROIMask,
#- create DRR's for both,
#- binarize the DRR of the ROIMask
#- compute its center of mass

def deformImageAndMaskAndComputeDRRs(img, ROIMask, deformation, projectionAngle=0, projectionAxis='Z', tryGPU=True, outputSize=[]):
    print('Start deformations and projections for deformation', deformation.name)
    image = deformation.deformImage(img, fillValue='closest', outputType=np.int16, tryGPU=tryGPU)
    # print(image.imageArray.shape, np.min(image.imageArray), np.max(image.imageArray), np.mean(image.imageArray))
    mask = deformation.deformImage(ROIMask, fillValue='closest', outputType=np.int16, tryGPU=tryGPU)
    centerOfMass3D = mask.centerOfMass

    DRR = forwardProjection(image, projectionAngle, axis=projectionAxis)
    DRRMask = forwardProjection(mask, projectionAngle, axis=projectionAxis)

    halfDiff = int((DRR.shape[1] - image.gridSize[2])/2)           ## not sure this will work if orientation is changed
    croppedDRR = DRR[:, halfDiff + 1:DRR.shape[1] - halfDiff - 1]         ## not sure this will work if orientation is changed
    croppedDRRMask = DRRMask[:, halfDiff + 1:DRRMask.shape[1] - halfDiff - 1] ## not sure this will work if orientation is changed

    if outputSize:
        # print('Before resampling')
        # print(croppedDRR.shape, np.min(croppedDRR), np.max(croppedDRR), np.mean(croppedDRR))
        ratio = [outputSize[0]/croppedDRR.shape[0], outputSize[1]/croppedDRR.shape[1]]
        croppedDRR = zoom(croppedDRR, ratio)
        croppedDRRMask = zoom(croppedDRRMask, ratio)
        # print('After resampling')
        # print(croppedDRR.shape, np.min(croppedDRR), np.max(croppedDRR), np.mean(croppedDRR))

    binaryDRRMask = getBinaryMaskFromROIDRR(croppedDRRMask)
    centerOfMass = get2DMaskCenterOfMass(binaryDRRMask)
    # print('CenterOfMass:', centerOfMass)

    del image  # to release the RAM
    del mask  # to release the RAM

    print('Deformations and projections finished for deformation', deformation.name)

    # plt.figure()
    # plt.subplot(1, 5, 1)
    # plt.imshow(DRR)
    # plt.subplot(1, 5, 2)
    # plt.imshow(croppedDRR)
    # plt.subplot(1, 5, 3)
    # plt.imshow(DRRMask)
    # plt.subplot(1, 5, 4)
    # plt.imshow(croppedDRRMask)
    # plt.subplot(1, 5, 5)
    # plt.imshow(binaryDRRMask)
    # plt.show()

    return [croppedDRR, binaryDRRMask, centerOfMass, centerOfMass3D]

#%%
# load the patient data structure and get the model and RTStruct

patient = loadDataStructure(dataPath)[0]
dynMod = patient.getPatientDataOfType("Dynamic3DModel")[0]
rtStruct = patient.getPatientDataOfType("RTStruct")[0]

#%%
# get the ROI and mask on which we want to apply the motion signal
print('Available ROIs')
rtStruct.print_ROINames()

gtvContour = rtStruct.getContourByName(otherContourToUse)
GTVMask = gtvContour.getBinaryMask(origin=dynMod.midp.origin, gridSize=dynMod.midp.gridSize, spacing=dynMod.midp.spacing)
gtvBox = getBoxAroundROI(GTVMask)

#%%
# get the body contour to adjust the crop in the direction of the DRR projection
bodyContour = rtStruct.getContourByName(bodyContourToUse)
bodyMask = bodyContour.getBinaryMask(origin=dynMod.midp.origin, gridSize=dynMod.midp.gridSize, spacing=dynMod.midp.spacing)
bodyBox = getBoxAroundROI(bodyMask)

if projAngle == 0 and projAxis == 'Z': # coronal
    croppingBox = [gtvBox[0], bodyBox[1], gtvBox[2]] ## create the used box combining the two boxes
elif projAngle == 90 and projAxis == 'Z': # sagittal
    croppingBox = [bodyBox[0], gtvBox[1], gtvBox[2]]
elif projAngle == 0 and projAxis == 'X': # coronal
    croppingBox = [gtvBox[0], bodyBox[1], gtvBox[2]]
elif projAngle == 0 and projAxis == 'Y': # sagittal
    croppingBox = [bodyBox[0], gtvBox[1], gtvBox[2]]
else:
    print('Do not know how to handle crop in this axis/angle configuration, so the body is used')
    croppingBox = [bodyBox[0], bodyBox[1], bodyBox[2]]

#%%
# crop the model data using the box
crop3DDataAroundBox(dynMod, croppingBox, marginInMM=marginInMM)

#%%
# if you want to see the crop in the opentps_core you can save the data in cropped version
saveSerializedObjects(patient, savingPath + 'croppedModelAndROIs')

#%%
# get the 3D center of mass of this ROI
gtvCenterOfMass = gtvContour.getCenterOfMass(dynMod.midp.origin, dynMod.midp.gridSize, dynMod.midp.spacing)
gtvCenterOfMassInVoxels = getVoxelIndexFromPosition(gtvCenterOfMass, dynMod.midp)
print('Used ROI name', gtvContour.name)
print('Used ROI center of mass :', gtvCenterOfMass)
print('Used ROI center of mass in voxels:', gtvCenterOfMassInVoxels)

if amplitude == 'model':
    ## to get amplitude from model !!! it takes some time because 10 displacement fields must be computed just for this
    modelValues = getAverageModelValuesAroundPosition(gtvCenterOfMass, dynMod, dimensionUsed='Z')
    amplitude = np.max(modelValues) - np.min(modelValues)
    print('Amplitude of deformation at ROI center of mass', amplitude)

#%%
# Signal creation
newSignal = SyntheticBreathingSignal(amplitude=amplitude,
                                        variationAmplitude=variationAmplitude,
                                        breathingPeriod=breathingPeriod,
                                        variationFrequency=variationFrequency,
                                        shift=shift,
                                        meanNoise=meanNoise,
                                        varianceNoise=varianceNoise,
                                        samplingPeriod=samplingPeriod,
                                        simulationTime=sequenceDurationInSecs,
                                        meanEvent=meanEvent)

newSignal.generate1DBreathingSignal()

pointList = [gtvCenterOfMass]
pointVoxelList = [gtvCenterOfMassInVoxels]
signalList = [newSignal]

saveSerializedObjects([signalList, pointList], savingPath + 'ROIsAndSignalObjects')
for signalIndex in range(len(signalList)):
    signalList[signalIndex] = signalList[signalIndex].breathingSignal

#%%
#show signals and ROIs

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.figure(figsize=(12, 6))
signalAx = plt.subplot(2, 1, 2)

for pointIndex, point in enumerate(pointList):
    ax = plt.subplot(2, 2 * len(pointList), 2 * pointIndex + 1)
    ax.set_title('Slice Y:' + str(pointVoxelList[pointIndex][1]))
    ax.imshow(np.rot90(dynMod.midp.imageArray[:, pointVoxelList[pointIndex][1], :]))
    ax.scatter([pointVoxelList[pointIndex][0]], [dynMod.midp.imageArray.shape[2] - pointVoxelList[pointIndex][2]],
                c=colors[pointIndex], marker="x", s=100)
    ax2 = plt.subplot(2, 2 * len(pointList), 2 * pointIndex + 2)
    ax2.set_title('Slice Z:' + str(pointVoxelList[pointIndex][2]))
    ax2.imshow(np.rot90(dynMod.midp.imageArray[:, :, pointVoxelList[pointIndex][2]]))
    ax2.scatter([pointVoxelList[pointIndex][0]], [dynMod.midp.imageArray.shape[2] - pointVoxelList[pointIndex][2]],
                c=colors[pointIndex], marker="x", s=100)
    signalAx.plot(newSignal.timestamps / 1000, signalList[pointIndex], c=colors[pointIndex])

signalAx.set_xlabel('Time (s)')
signalAx.set_ylabel('Deformation amplitude in Z direction (mm)')
plt.savefig(savingPath + 'ROI_And_Signals_fig.pdf', dpi=300)
plt.show()

## -------------------------------------------------------------

sequenceSize = newSignal.breathingSignal.shape[0]
print('Sequence Size =', sequenceSize, 'split by stack of ', subSequenceSize, '. Multiprocessing =', multiprocessing)

subSequencesIndexes = [subSequenceSize * i for i in range(math.ceil(sequenceSize/subSequenceSize))]
subSequencesIndexes.append(sequenceSize)
print('Sub sequences indexes', subSequencesIndexes)

startTime = time.time()

if multiprocessing == False:

    resultList = []

    for i in range(len(subSequencesIndexes)-1):
        print('Creating deformations for images', subSequencesIndexes[i], 'to', subSequencesIndexes[i + 1] - 1)

        deformationList = generateDeformationListFromBreathingSignalsAndModel(dynMod,
                                                                                signalList,
                                                                                pointList,
                                                                                signalIdxUsed=[subSequencesIndexes[i], subSequencesIndexes[i+1]],
                                                                                dimensionUsed='Z',
                                                                                outputType=np.float32)

        for deformationIndex, deformation in enumerate(deformationList):
            resultList.append(deformImageAndMaskAndComputeDRRs(dynMod.midp,
                                                                GTVMask,
                                                                deformation,
                                                                projectionAngle=projAngle,
                                                                projectionAxis=projAxis,
                                                                outputSize=outputSize,
                                                                tryGPU=True))


    savingPath += dataSetDataFolder + f'Patient_0_{sequenceSize}_DRRMasksAndCOM'
    saveSerializedObjects(resultList, savingPath + str(sequenceSize))


elif multiprocessing == True:

    resultList = []

    if subSequenceSize > maxMultiProcUse:  ## re-adjust the subSequenceSize since this will be done in multi processing
        subSequenceSize = maxMultiProcUse
        print('SubSequenceSize put to', maxMultiProcUse, 'for multiprocessing.')
        print('Sequence Size =', sequenceSize, 'split by stack of ', subSequenceSize, '. Multiprocessing =', multiprocessing)
        subSequencesIndexes = [subSequenceSize * i for i in range(math.ceil(sequenceSize / subSequenceSize))]
        subSequencesIndexes.append(sequenceSize)

    for i in range(len(subSequencesIndexes)-1):
        print('Creating deformations for images', subSequencesIndexes[i], 'to', subSequencesIndexes[i + 1] - 1)

        deformationList = generateDeformationListFromBreathingSignalsAndModel(dynMod,
                                                                                signalList,
                                                                                pointList,
                                                                                signalIdxUsed=[subSequencesIndexes[i], subSequencesIndexes[i+1]],
                                                                                dimensionUsed='Z',
                                                                                outputType=np.float32)

        print('Start multi process deformation with', len(deformationList), 'deformations')
        with concurrent.futures.ProcessPoolExecutor() as executor:

            results = executor.map(deformImageAndMaskAndComputeDRRs, repeat(dynMod.midp), repeat(GTVMask), deformationList, repeat(projAngle), repeat(projAxis), repeat(tryGPU), repeat(outputSize))
            resultList += results

        print('ResultList lenght', len(resultList))

    savingPath += dataSetDataFolder + f'Patient_0_{sequenceSize}_DRRMasksAndCOM_multiProcTest'
    saveSerializedObjects(resultList, savingPath)

stopTime = time.time()
print('Test with multiprocessing =', multiprocessing, '. Sub-sequence size:', str(subSequenceSize), 'finished in', np.round(stopTime - startTime, 2) / 60, 'minutes')
print(np.round((stopTime - startTime)/len(resultList), 2), 'sec per sample')
