'''
Deformation from Weight Maps
=========================
author: OpenTPS team

This example demonstrates how to apply a deformation to a model using weight maps and visualize the results.

running time: ~ 7 minutes
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

import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import os

#%%
#import the needed opentps.core packages

from opentps.core.processing.imageProcessing import resampler3D
from opentps.core.data.dynamicData._dynamic3DModel import Dynamic3DModel
from opentps.core.data.dynamicData._dynamic3DSequence import Dynamic3DSequence
from opentps.core.data.images import CTImage
from opentps.core.data.images import ROIMask
from opentps.core.processing.deformableDataAugmentationToolBox.weightMaps import generateDeformationFromTrackers, generateDeformationFromTrackersAndWeightMaps

logger = logging.getLogger(__name__)

#%%
# Output path
#------------
output_path = os.path.join(os.getcwd(), 'Output', 'ExampleDeformationFromWeightMaps')
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
# Generate synthetic 4DCT and MidP
#---------------------------------

# GENERATE SYNTHETIC 4D INPUT SEQUENCE
CT4D = createSynthetic4DCT()

# CREATE TRACKER POSITIONS
trackers = [[30, 75, 40],
            [70, 75, 40],
            [100, 75, 40],
            [140, 75, 40]]

# GENERATE MIDP
Model4D = Dynamic3DModel()
Model4D.computeMidPositionImage(CT4D, 0, tryGPU=True)

# GENERATE ADDITIONAL PHASES
df1, wm = generateDeformationFromTrackers(Model4D, [0, 0, 2/4, 2/4], [1, 1, 1, 1], trackers)
im1 = df1.deformImage(Model4D.midp, fillValue='closest')
df2, wm = generateDeformationFromTrackers(Model4D, [0.5/4, 0.5/4, 1.5/4, 1.5/4], [1, 1, 1, 1], trackers)
im2 = df2.deformImage(Model4D.midp, fillValue='closest')
df3 = generateDeformationFromTrackersAndWeightMaps(Model4D, [0, 0, 2/4, 2/4], [2, 2, 2, 2], wm)
im3 = df3.deformImage(Model4D.midp, fillValue='closest')

# RESAMPLE WEIGHT MAPS TO IMAGE RESOLUTION
for i in range(len(trackers)):
    resampler3D.resampleImage3DOnImage3D(wm[i], Model4D.midp, inPlace=True, fillValue=-1024.)

#%%
# Display results
#----------------
fig, ax = plt.subplots(2, 5)
ax[0,0].imshow(Model4D.midp.imageArray[:, 50, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
s0 = wm[0].imageArray[:, 50, :].T[::-1, ::1]
s1 = wm[1].imageArray[:, 50, :].T[::-1, ::1]
s2 = wm[2].imageArray[:, 50, :].T[::-1, ::1]
s3 = wm[3].imageArray[:, 50, :].T[::-1, ::1]
ax[0,1].imshow(s0, cmap='Reds', origin='upper', vmin=0, vmax=1)
ax[0,2].imshow(s1, cmap='Reds', origin='upper', vmin=0, vmax=1)
ax[0,3].imshow(s2, cmap='Blues', origin='upper', vmin=0, vmax=1)
ax[0,4].imshow(s3, cmap='Blues', origin='upper', vmin=0, vmax=1)
ax[0,0].plot(trackers[0][0],100-trackers[0][2],'ro')
ax[0,0].plot(trackers[1][0],100-trackers[1][2],'ro')
ax[0,0].plot(trackers[2][0],100-trackers[2][2],'bo')
ax[0,0].plot(trackers[3][0],100-trackers[3][2],'bo')
ax[0,1].plot(trackers[0][0],100-trackers[0][2],'ro')
ax[0,2].plot(trackers[1][0],100-trackers[1][2],'ro')
ax[0,3].plot(trackers[2][0],100-trackers[2][2],'bo')
ax[0,4].plot(trackers[3][0],100-trackers[3][2],'bo')

ax[1,0].imshow(Model4D.midp.imageArray[:, :, 50].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
s0 = wm[0].imageArray[:, :, 50].T[::-1, ::1]
s1 = wm[1].imageArray[:, :, 50].T[::-1, ::1]
s2 = wm[2].imageArray[:, :, 50].T[::-1, ::1]
s3 = wm[3].imageArray[:, :, 50].T[::-1, ::1]
ax[1,1].imshow(s0, cmap='Reds', origin='upper', vmin=0, vmax=1)
ax[1,2].imshow(s1, cmap='Reds', origin='upper', vmin=0, vmax=1)
ax[1,3].imshow(s2, cmap='Blues', origin='upper', vmin=0, vmax=1)
ax[1,4].imshow(s3, cmap='Blues', origin='upper', vmin=0, vmax=1)
ax[1,0].plot(trackers[0][0],trackers[0][1],'ro')
ax[1,0].plot(trackers[1][0],trackers[1][1],'ro')
ax[1,0].plot(trackers[2][0],trackers[2][1],'bo')
ax[1,0].plot(trackers[3][0],trackers[3][1],'bo')
ax[1,1].plot(trackers[0][0],trackers[0][1],'ro')
ax[1,2].plot(trackers[1][0],trackers[1][1],'ro')
ax[1,3].plot(trackers[2][0],trackers[2][1],'bo')
ax[1,4].plot(trackers[3][0],trackers[3][1],'bo')
ax[0,0].title.set_text('MidP and trackers')
ax[0,1].title.set_text('Tracker 1')
ax[0,2].title.set_text('Tracker 2')
ax[0,3].title.set_text('Tracker 3')
ax[0,4].title.set_text('Tracker 4')

fig, ax = plt.subplots(2, 4)
fig.tight_layout()
y_slice = round(Model4D.midp.imageArray.shape[1]/2)-1
ax[0,0].imshow(CT4D.dyn3DImageList[0].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0,0].title.set_text('Phase 0')
ax[0,1].imshow(CT4D.dyn3DImageList[1].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0,1].title.set_text('Phase 1')
ax[0,2].imshow(CT4D.dyn3DImageList[2].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0,2].title.set_text('Phase 2')
ax[0,3].imshow(CT4D.dyn3DImageList[3].imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[0,3].title.set_text('Phase 3')
ax[1,0].imshow(Model4D.midp.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1,0].imshow(wm[0].imageArray[:, y_slice, :].T[::-1, ::1] + wm[1].imageArray[:, y_slice, :].T[::-1, ::1], cmap='Reds', origin='upper', vmin=0, vmax=1, alpha=0.3)
ax[1,0].imshow(wm[2].imageArray[:, y_slice, :].T[::-1, ::1] + wm[3].imageArray[:, y_slice, :].T[::-1, ::1], cmap='Blues', origin='upper', vmin=0, vmax=1, alpha=0.3)
ax[1, 0].plot(trackers[0][0],100-trackers[0][2], 'ro')
ax[1, 0].plot(trackers[1][0],100-trackers[1][2], 'ro')
ax[1, 0].plot(trackers[2][0],100-trackers[2][2], 'bo')
ax[1, 0].plot(trackers[3][0],100-trackers[3][2], 'bo')
ax[1,0].title.set_text('MidP and weight maps')
ax[1,1].imshow(im1.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1,1].title.set_text('phases [0,2] - amplitude 1')
ax[1,2].imshow(im2.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1,2].title.set_text('phases [0.5,1.5] - amplitude 1')
ax[1,3].imshow(im3.imageArray[:, y_slice, :].T[::-1, ::1], cmap='gray', origin='upper', vmin=-1000, vmax=1000)
ax[1,3].title.set_text('phases [0,2] - amplitude 2')

plt.savefig(os.path.join(output_path, 'DeformationFromWeightMaps.png'))
print('done')
print(' ')
plt.show()
