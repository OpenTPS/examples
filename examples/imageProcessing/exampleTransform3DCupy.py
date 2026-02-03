'''
Transform 3D Image with CuPy
============================
author: OpenTPS team

This example demonstrates how to apply a 3D transformation to a synthetic CT image using the OpenTPS library with CuPy for efficient computation.

running time: ~ 5 minutes
'''
#%% 
# Setting up the environment in google collab
#--------------------------------------------
# You will first need to change the execution parameters in your Colab environment (bottom left) from CPU to GPU so that you can run the example.
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
import copy

import matplotlib.pyplot as plt
import logging
import os

#%%
#import the needed opentps.core packages
from opentps.core.data.images import VectorField3D
from opentps.core.data.dynamicData._dynamic3DModel import Dynamic3DModel
from opentps.core.data._transform3D import Transform3D
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from showStuff import showModelWithAnimatedFields
from syntheticData import *
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
from opentps.core.processing.imageProcessing.imageTransform3D import rotateData, translateData
from opentps.core.processing.imageProcessing.resampler3D import resample

logger = logging.getLogger(__name__)

#%%
# Output path
#-------------
output_path = os.path.join(os.getcwd(), 'Output', 'ExampleTransform3DCupy')
if not os.path.exists(output_path):
        os.makedirs(output_path)
logger.info('Files will be stored in {}'.format(output_path))

#%%
# GENERATE SYNTHETIC INPUT IMAGES
#--------------------------------
fixed = CTImage()
fixed.spacing = np.array([1, 1, 1])
fixed.imageArray = np.full((20, 20, 20), -1000)
fixed.imageArray[11:16, 5:14, 11:14] = 100.0

moving = copy.deepcopy(fixed)
movingTrans = copy.deepcopy(fixed)
movingRot = copy.deepcopy(fixed)
movingBoth = copy.deepcopy(fixed)

translation = np.array([0, 0, 0])
rotation = np.array([0, 45, 0])
rotCenter='imgCenter'

#%%
# Create a transform 3D
#----------------------
print('Create a transform 3D')
transform3D = Transform3D()
transform3D.initFromTranslationAndRotationVectors(transVec=translation, rotVec=rotation)
transform3D.setCenter(rotCenter)
print('Translation', transform3D.getTranslation())
print('Rotation', transform3D.getRotationAngles(inDegrees=True))

print('moving with transform3D')
moving = transform3D.deformData(moving, outputBox='same', fillValue=-1000, tryGPU=True)

print('moving translation')
translateData(movingTrans, translationInMM=translation, outputBox='same', fillValue=-1000, tryGPU=True)
print('moving rotation')
rotateData(movingRot, rotAnglesInDeg=rotation, rotCenter=rotCenter, outputBox='same', fillValue=-1000, tryGPU=True)
# movingRot = resampleImage3DOnImage3D(movingRot, fixedImage=fixed, fillValue=-1000)
print('moving both')
translateData(movingBoth, translationInMM=translation, outputBox='same', fillValue=-1000, tryGPU=True)
rotateData(movingBoth, rotAnglesInDeg=rotation, rotCenter=rotCenter, outputBox='same', fillValue=-1000, tryGPU=True)

y_slice = 10

fig, ax = plt.subplots(1, 6)
ax[0].set_title('fixed')
ax[0].imshow(fixed.imageArray[:, y_slice, :])
ax[0].set_xlabel(f"{fixed.origin}\n{fixed.spacing}\n{fixed.gridSize}")

ax[1].set_title('translateData')
ax[1].imshow(movingTrans.imageArray[:, y_slice, :])
ax[1].set_xlabel(f"{movingTrans.origin}\n{movingTrans.spacing}\n{movingTrans.gridSize}")

ax[2].set_title('rotateData')
ax[2].imshow(movingRot.imageArray[:, y_slice, :])
ax[2].set_xlabel(f"{movingRot.origin}\n{movingRot.spacing}\n{movingRot.gridSize}")

ax[3].set_title('both')
ax[3].imshow(movingBoth.imageArray[:, y_slice, :])
ax[3].set_xlabel(f"{movingBoth.origin}\n{movingBoth.spacing}\n{movingBoth.gridSize}")

ax[4].set_title('transform3D')
ax[4].imshow(moving.imageArray[:, y_slice, :])
ax[4].set_xlabel(f"{moving.origin}\n{moving.spacing}\n{moving.gridSize}")

ax[5].set_title('transform3D-both')
ax[5].imshow(moving.imageArray[:, y_slice, :] - movingBoth.imageArray[:, y_slice, :])

plt.savefig(os.path.join(output_path, 'ExampleTransform3DCupy.png'))
plt.show()

#%%
# Create a dynamic model with the transform
#------------------------------------------
print(' --------------------- start test with model -----------------------------')

CT4D = createSynthetic4DCT(numberOfPhases=4)
# GENERATE MIDP
fixedDynMod = Dynamic3DModel()
fixedDynMod.computeMidPositionImage(CT4D, 0, tryGPU=True)

print(fixedDynMod.midp.origin, fixedDynMod.midp.spacing, fixedDynMod.midp.gridSize)
print('Resample model image')
fixedDynMod = resample(fixedDynMod, gridSize=(80, 50, 50))
print('after resampling', fixedDynMod.midp.origin, fixedDynMod.midp.spacing, fixedDynMod.midp.gridSize)

# option 3
for field in fixedDynMod.deformationList:
    print('Resample model field')
    field.resample(spacing=fixedDynMod.midp.spacing, gridSize=fixedDynMod.midp.gridSize, origin=fixedDynMod.midp.origin)
    print('after resampling', field.origin, field.spacing, field.gridSize)

showModelWithAnimatedFields(fixedDynMod)

movingDynMod = copy.copy(fixedDynMod)

rotateData(movingDynMod, rotAnglesInDeg=rotation, rotCenter=rotCenter, outputBox='same', fillValue=-1000, tryGPU=True)

showModelWithAnimatedFields(movingDynMod)

#%%
#Generate synthetic input images 
#-------------------------------
fixed = CTImage()
fixed.imageArray = np.full((20, 20, 20), -1000)
y_slice = 10

pointList = [[15, y_slice, 15], [15, y_slice, 10], [12, y_slice, 12], [10, y_slice, 10]]
for point in pointList:
    fixed.imageArray[point[0], point[1], point[2]] = 200

fieldFixed = VectorField3D()
fieldFixed.imageArray = np.zeros((20, 20, 20, 3))
vectorList = [np.array([2, 3, 4]), np.array([0, 3, 4]), np.array([7, 3, 3]), np.array([2, 0, 0])]
for pointIdx in range(len(pointList)):
    fieldFixed.imageArray[pointList[pointIdx][0], pointList[pointIdx][1], pointList[pointIdx][2]] = vectorList[
        pointIdx]

moving = copy.copy(fixed)
fieldMoving = copy.copy(fieldFixed)


#%%
# Create a transform 3D
#----------------------
print('Create a transform 3D')
transform3D = Transform3D()
transform3D.initFromTranslationAndRotationVectors(transVec=translation, rotVec=rotation)
transform3D.setCenter(rotCenter)
print('Translation', transform3D.getTranslation())
print('Rotation', transform3D.getRotationAngles(inDegrees=True))

print('moving with transform3D')
# moving = transform3D.deformData(moving, fillValue=-1000, outputBox='same', tryGPU=True)
# fieldMoving = transform3D.deformData(fieldMoving, fillValue=0, outputBox='same', tryGPU=True)

rotateData(moving, rotAnglesInDeg=rotation, rotCenter=rotCenter, outputBox='same', fillValue=-1000, tryGPU=True)
rotateData(fieldMoving, rotAnglesInDeg=rotation, rotCenter=rotCenter, outputBox='same', fillValue=0, tryGPU=True)

moving = resampleImage3DOnImage3D(moving, fixedImage=fixed, fillValue=-1000)
print('fixed.origin', fixed.origin, 'moving.origin', moving.origin)
fieldMoving = resampleImage3DOnImage3D(fieldMoving, fixedImage=fixed, fillValue=0)
print('fieldFixed.origin', fieldFixed.origin, 'fieldMoving.origin', fieldMoving.origin)

print('ici ', fieldMoving.imageArray[10, y_slice, 10])

compXFixed = fieldFixed.imageArray[:, y_slice, :, 0]
compZFixed = fieldFixed.imageArray[:, y_slice, :, 2]
compXMoving = fieldMoving.imageArray[:, y_slice, :, 0]
compZMoving = fieldMoving.imageArray[:, y_slice, :, 2]

#%%
# Display results
#----------------

fig, ax = plt.subplots(1, 2)
ax[0].imshow(fixed.imageArray[:, y_slice, :])
ax[0].quiver(compZFixed, compXFixed, alpha=0.5, color='red', angles='xy', scale_units='xy', scale=2, width=.010)
ax[1].imshow(moving.imageArray[:, y_slice, :])
ax[1].quiver(compZMoving, compXMoving, alpha=0.5, color='green', angles='xy', scale_units='xy', scale=2, width=.010)
plt.show()


print('start ROIMask test')
fixedMask = ROIMask.fromImage3D(fixed)
fixedMask.imageArray = np.zeros(fixedMask.gridSize).astype(bool)
fixedMask.imageArray[12:15, 8:12, 8:18] = True
plt.figure()
plt.imshow(fixedMask.imageArray[:, y_slice, :])
plt.show()

print(fixedMask.origin, fixedMask.gridSize, fixedMask.spacing, fixedMask.imageArray.dtype)

movingMask = copy.copy(fixedMask)
movingMask = transform3D.deformData(movingMask, outputBox='same')

print(movingMask.origin, movingMask.gridSize, movingMask.spacing, movingMask.imageArray.dtype)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(fixedMask.imageArray[:, y_slice, :])
plt.savefig(os.path.join(output_path, 'ExampleTransform3DCupy_fixexMask.png'))
plt.subplot(1, 2, 2)
plt.imshow(movingMask.imageArray[:, y_slice, :])
plt.savefig(os.path.join(output_path, 'ExampleTransform3DCupy_movingMask.png'))
plt.show()