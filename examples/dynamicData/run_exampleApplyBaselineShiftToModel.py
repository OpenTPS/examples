'''
Applying Baseline Shift to a Model
==================================
author: OpenTPS team

This example demonstrates how to apply a baseline shift to a model and visualize the results.

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
    get_ipython().system('pip install scipy==1.10.1')
    import opentps

#%%
#imports

import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import os

#%%
#import the needed opentps.core packages

from opentps.core.processing.imageProcessing.syntheticDeformation import applyBaselineShift
from opentps.core.data.dynamicData._dynamic3DModel import Dynamic3DModel
from opentps.core.data.dynamicData._dynamic3DSequence import Dynamic3DSequence
from opentps.core.data.images import CTImage
from opentps.core.data.images import ROIMask

logger = logging.getLogger(__name__)

#%%
# Output path
#------------
output_path = os.path.join(os.getcwd(), 'Output', 'ExampleApplyBaselineShiftToModel')
if not os.path.exists(output_path):
        os.makedirs(output_path)
logger.info('Files will be stored in {}'.format(output_path))

#%%
# Synthetic 4DCT generation function
#----------------------------------------
def getPhasesPositions(numberOfPhases, minValue, maxValue):

    angleList = np.linspace(0, 2 * math.pi, numberOfPhases + 1)[:-1]
    cosList = np.cos(angleList)

    diff = maxValue - minValue

    posList = minValue + diff / 2 + cosList * diff / 2

    return posList.astype(np.uint8)

def createSynthetic3DCT(diaphragmPos = 20, targetPos = [50, 100, 35], spacing=[1, 1, 2], returnTumorMask = False):
    # GENERATE SYNTHETIC CT IMAGE
    # background
    im = np.full((170, 170, 100), -1000)
    im[20:150, 70:130, :] = 0
    # left lung
    im[30:70, 80:120, diaphragmPos:] = -800
    # right lung
    im[100:140, 80:120, diaphragmPos:] = -800
    # target
    im[targetPos[0]-5:targetPos[0]+5, targetPos[1]-5:targetPos[1]+5, targetPos[2]-5:targetPos[2]+5] = 0
    # vertebral column
    im[80:90, 95:105, :] = 800
    # rib
    im[22:26, 90:110, 46:50] = 800
    # couch
    im[:, 130:135, :] = 100
    ct = CTImage(imageArray=im, name='fixed', origin=[0, 0, 0], spacing=spacing)

    if returnTumorMask:
        mask = np.full((170, 170, 100), 0)
        mask[targetPos[0]-5:targetPos[0]+5, targetPos[1]-5:targetPos[1]+5, targetPos[2]-5:targetPos[2]+5] = 1
        roi = ROIMask(imageArray=mask, origin=[0, 0, 0], spacing=spacing)

        return ct, roi

    else:
        return ct


def createSynthetic4DCT(numberOfPhases=4, spacing=[1, 1, 2], returnTumorMasks=False, motionNoise=True):

    # GENERATE SYNTHETIC 4D INPUT SEQUENCE
    CT4D = Dynamic3DSequence()

    ## For the diaphragm position
    diaphMotionAmp = 12
    diaphMinPos = 20
    diaphPosList = getPhasesPositions(numberOfPhases, diaphMinPos, diaphMinPos+diaphMotionAmp)

    if motionNoise:
        diaphNoise = [[3, 1],
                  [6, -1],
                  [9, -1],
                  [12, 1],
                  [15, 1]]
    else:
        diaphNoise = [[3, 0],
                      [6, 0],
                      [9, 0],
                      [12, 0],
                      [15, 0]]

    for elemIdx in range(len(diaphNoise)):
        if diaphNoise[elemIdx][0] <= numberOfPhases - 1:
            diaphPosList[diaphNoise[elemIdx][0]] += diaphNoise[elemIdx][1]

    ## For the target z position
    zMotionAmp = int(np.round(diaphMotionAmp * 0.8))
    zMinPos = 40
    zPosList = getPhasesPositions(numberOfPhases, zMinPos, zMinPos+zMotionAmp)

    if motionNoise:
        zNoise = [[3, 1],
                  [6, -1],
                  [9, -1],
                  [12, 1],
                  [15, 1]]
    else:
        zNoise = [[3, 0],
                      [6, 0],
                      [9, 0],
                      [12, 0],
                      [15, 0]]

    for elemIdx in range(len(zNoise)):
        if zNoise[elemIdx][0] <= numberOfPhases - 1:
            zPosList[zNoise[elemIdx][0]] += zNoise[elemIdx][1]

    ## For the target x position
    xMotionAmp = 6
    xMinPos = 42
    xPosList = getPhasesPositions(numberOfPhases, xMinPos, xMinPos+xMotionAmp)

    if motionNoise:
        xNoise = [[3, 1],
                  [6, -1],
                  [9, -1],
                  [12, 1],
                  [15, 1]]
    else:
        xNoise = [[3, 0],
                  [6, 0],
                  [9, 0],
                  [12, 0],
                  [15, 0]]

    for elemIdx in range(len(xNoise)):
        if xNoise[elemIdx][0] <= numberOfPhases-1:
            xPosList[xNoise[elemIdx][0]] += xNoise[elemIdx][1]

    xPosList = np.roll(xPosList, 2)
    # print('xPosList', xPosList)

    phaseList = []
    if returnTumorMasks:
        maskList = []
        for phaseIndex in range(numberOfPhases):
            phase,  mask = createSynthetic3DCT(targetPos=[xPosList[phaseIndex], 95, zPosList[phaseIndex]], diaphragmPos=diaphPosList[phaseIndex], spacing=spacing, returnTumorMask=returnTumorMasks)
            phaseList.append(phase)
            maskList.append(mask)

    else:
        for phaseIndex in range(numberOfPhases):
            phase = createSynthetic3DCT(targetPos=[xPosList[phaseIndex], 95, zPosList[phaseIndex]], diaphragmPos=diaphPosList[phaseIndex], spacing=spacing)
            phaseList.append(phase)

    CT4D.dyn3DImageList = phaseList
    if returnTumorMasks:
        return CT4D, maskList
    else:
        return CT4D

#%%
# Generate synthetic 4DCT, mask, and MidP
#----------------------------------------


# GENERATE SYNTHETIC 4DCT
CT4D = createSynthetic4DCT()

# GENERATE MASK
mask = np.full(CT4D.dyn3DImageList[0].gridSize, 0)
mask[38:52, 87:103, 39:54] = 1
roi = ROIMask(imageArray=mask, origin=[0, 0, 0], spacing=[1, 1, 1.5])

# GENERATE MIDP
Model = Dynamic3DModel()
Model.computeMidPositionImage(CT4D, 0, tryGPU=True)

#%%
# Apply baseline shift
#---------------------

ModelShifted, maskShifted = applyBaselineShift(Model, roi, [5, 0, 10])

#%%
# Regenerate 4D sequences from models
#------------------------------------

CT4DRegen = Dynamic3DSequence()
for i in range(len(CT4D.dyn3DImageList)):
    CT4DRegen.dyn3DImageList.append(Model.generate3DImage(i / len(CT4D.dyn3DImageList), amplitude=1))
CT4DShifted = Dynamic3DSequence()
for i in range(len(CT4D.dyn3DImageList)):
    CT4DShifted.dyn3DImageList.append(ModelShifted.generate3DImage(i/len(CT4D.dyn3DImageList), amplitude=1))

#%%
# Display results
#----------------
fig, ax = plt.subplots(3, 7)
fig.tight_layout()
y_slice = 95
ax[1, 0].imshow(Model.midp.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1, 0].title.set_text('MidP')
ax[2, 0].imshow(ModelShifted.midp.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[2, 0].title.set_text('MidP shifted')

average = CT4D.dyn3DImageList[0].copy()
for i in range(len(CT4D.dyn3DImageList)-1):
    average._imageArray += CT4D.dyn3DImageList[i+1]._imageArray
average._imageArray = average.imageArray/len(CT4D.dyn3DImageList)
averageRegen = CT4DRegen.dyn3DImageList[0].copy()
for i in range(len(CT4DRegen.dyn3DImageList) - 1):
    averageRegen._imageArray += CT4DRegen.dyn3DImageList[i + 1]._imageArray
averageRegen._imageArray = averageRegen.imageArray / len(CT4DRegen.dyn3DImageList)
averageShifted = CT4DShifted.dyn3DImageList[0].copy()
for i in range(len(CT4DShifted.dyn3DImageList) - 1):
    averageShifted._imageArray += CT4DShifted.dyn3DImageList[i + 1]._imageArray
averageShifted._imageArray = averageShifted.imageArray / len(CT4DShifted.dyn3DImageList)

ax[0, 1].imshow(average.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0, 1].title.set_text('Average')
ax[1, 1].imshow(averageRegen.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1, 1].title.set_text('Gen average')
ax[2, 1].imshow(averageShifted.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[2, 1].title.set_text('Gen average shifted')

averageRegen._imageArray -= average._imageArray
averageShifted._imageArray -= average._imageArray
average._imageArray -= average._imageArray
ax[0, 0].imshow(average.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0, 0].title.set_text('-')
ax[0, 2].imshow(average.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0, 2].title.set_text('-')
ax[1, 2].imshow(averageRegen.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1, 2].title.set_text('Gen average diff')
ax[2, 2].imshow(averageShifted.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[2, 2].title.set_text('Gen average shifted diff')

ax[0, 3].imshow(CT4D.dyn3DImageList[0].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0, 3].title.set_text('Phase 0')
ax[1, 3].imshow(CT4DRegen.dyn3DImageList[0].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1, 3].title.set_text('Gen phase 0')
ax[2, 3].imshow(CT4DShifted.dyn3DImageList[0].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[2, 3].title.set_text('Gen phase 0 shifted')

ax[0, 4].imshow(CT4D.dyn3DImageList[1].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0, 4].title.set_text('Phase 1')
ax[1, 4].imshow(CT4DRegen.dyn3DImageList[1].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1, 4].title.set_text('Gen phase 1')
ax[2, 4].imshow(CT4DShifted.dyn3DImageList[1].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[2, 4].title.set_text('Gen phase 1 shifted')

ax[0, 5].imshow(CT4D.dyn3DImageList[2].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0, 5].title.set_text('Phase 2')
ax[1, 5].imshow(CT4DRegen.dyn3DImageList[2].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1, 5].title.set_text('Gen phase 2')
ax[2, 5].imshow(CT4DShifted.dyn3DImageList[2].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[2, 5].title.set_text('Gen phase 2 shifted')

ax[0, 6].imshow(CT4D.dyn3DImageList[3].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0, 6].title.set_text('Phase 3')
ax[1, 6].imshow(CT4DRegen.dyn3DImageList[3].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1, 6].title.set_text('Gen phase 3')
ax[2, 6].imshow(CT4DShifted.dyn3DImageList[3].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[2, 6].title.set_text('Gen phase 3 shifted')

plt.savefig(os.path.join(output_path, 'BaselinesSHift.png')) 
print('done')
plt.show()

