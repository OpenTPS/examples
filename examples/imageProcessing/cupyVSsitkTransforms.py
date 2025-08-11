'''
Cupy vs SitkTransforms
=========================
author: OpenTPS team


'''
# %% [skip]
# installing opentps in colab
import sys
if "google.colab" in sys.modules:
    from IPython import get_ipython
    get_ipython().system('git clone https://gitlab.com/openmcsquare/opentps.git')
    get_ipython().system('pip install ./opentps')
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
from opentps.core.examples.showStuff import showModelWithAnimatedFields
from opentps.core.examples.syntheticData import *
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
from opentps.core.processing.imageProcessing.imageTransform3D import rotateData, translateData
from opentps.core.processing.imageProcessing.resampler3D import resample

logger = logging.getLogger(__name__)
#%%
# Output path
# -----------
output_path = os.path.join(os.getcwd(), 'Output', 'CupyVsSitk')
if not os.path.exists(output_path):
        os.makedirs(output_path)
logger.info('Files will be stored in {}'.format(output_path))

#%%
# Generate synthetic input immages
# ---------------------------------

imgSize = [40, 40, 40]
imgSpacing = [1, 1, 2]
objectBorder = [[21, 33], [int(imgSize[1]/4), 3*int(imgSize[1]/4)], [21, 34]]

translation = np.array([-10.22, 0, -14.56])
rotation = np.array([0, 30, 0])
rotCenter = 'imgCenter'
outputBox = 'same'
# interpOrder = 1

showImage = True
showField = True
showMask = True

fixed = CTImage()
fixed.spacing = np.array(imgSpacing)
fixed.imageArray = np.full(imgSize, -1000)
fixed.imageArray[objectBorder[0][0]: objectBorder[0][1],
                objectBorder[1][0]: objectBorder[1][1],
                objectBorder[2][0]: objectBorder[2][1]] = 100.0

y_slice = int(imgSize[1]/2)
pointList = [[objectBorder[0][0], y_slice, objectBorder[2][1]-1],
                [objectBorder[0][1]-1, y_slice, objectBorder[2][0]],
                [objectBorder[0][0]+1, y_slice, objectBorder[2][0]+1],
                [objectBorder[0][0], y_slice, objectBorder[2][0]]]

fieldFixed = VectorField3D()
fieldFixed.imageArray = np.zeros((imgSize[0], imgSize[1], imgSize[2], 3))
fieldFixed.spacing = np.array(imgSpacing)
vectorList = [np.array([4, 6, 8]), np.array([0, 6, 8]), np.array([14, 6, 6]), np.array([6, 0, 0])]
for pointIdx in range(len(pointList)):
    fieldFixed.imageArray[pointList[pointIdx][0], pointList[pointIdx][1], pointList[pointIdx][2]] = vectorList[
        pointIdx]

maskFixed = ROIMask()
maskFixed.spacing = np.array(imgSpacing)
maskFixed.imageArray = np.zeros(imgSize).astype(bool)
maskFixed.imageArray[objectBorder[0][0]: objectBorder[0][1],
                    objectBorder[1][0]: objectBorder[1][1],
                    objectBorder[2][0]: objectBorder[2][1]] = True

#%%
# this function is just to see the results
def showImagesAndFieldAndMask(fixed, movingCupy1, movingCupy3, movingSitk, fieldFixed, fieldMovingCupy1, fieldMovingCupy3, fieldMovingSitk,
                                maskFixed, maskMovingCupy1, maskMovingCupy3, maskMovingSitk, y_slice, figTitle, showImage=True,
                                showField=True, showMask=True, ):

    compXFixed = fieldFixed.imageArray[:, y_slice, :, 0]
    compZFixed = fieldFixed.imageArray[:, y_slice, :, 2]
    compXMovingCupy1 = fieldMovingCupy1.imageArray[:, y_slice, :, 0]
    compZMovingCupy1 = fieldMovingCupy1.imageArray[:, y_slice, :, 2]
    compXMovingCupy3 = fieldMovingCupy3.imageArray[:, y_slice, :, 0]
    compZMovingCupy3 = fieldMovingCupy3.imageArray[:, y_slice, :, 2]
    compXMovingSITK = fieldMovingSitk.imageArray[:, y_slice, :, 0]
    compZMovingSITK = fieldMovingSitk.imageArray[:, y_slice, :, 2]

    fig, ax = plt.subplots(3, 4)
    fig.suptitle(figTitle)

    if showImage:
        ax[0, 0].imshow(fixed.imageArray[:, y_slice, :])
        ax[0, 1].imshow(movingCupy1.imageArray[:, y_slice, :])
        ax[0, 2].imshow(movingCupy3.imageArray[:, y_slice, :])
        ax[0, 3].imshow(movingSitk.imageArray[:, y_slice, :])
        ax[1, 0].imshow(movingCupy1.imageArray[:, y_slice, :] - movingSitk.imageArray[:, y_slice, :])
        ax[1, 0].set_xlabel('Img diff cupy1-sitk')
        ax[1, 1].imshow(movingCupy1.imageArray[:, y_slice, :] - movingCupy3.imageArray[:, y_slice, :])
        ax[1, 1].set_xlabel('Img diff cupy1-cupy3')
    if showField:
        ax[0, 0].quiver(compZFixed, compXFixed, alpha=0.5, color='red', angles='xy', scale_units='xy', scale=2, width=.010)
        ax[0, 1].quiver(compZMovingCupy1, compXMovingCupy1, alpha=0.5, color='green', angles='xy', scale_units='xy', scale=2, width=.010)
        ax[0, 2].quiver(compZMovingCupy3, compXMovingCupy3, alpha=0.5, color='green', angles='xy', scale_units='xy', scale=2, width=.010)
        ax[0, 3].quiver(compZMovingSITK, compXMovingSITK, alpha=0.5, color='green', angles='xy', scale_units='xy', scale=2, width=.010)
        ax[1, 2].quiver(compZMovingCupy1 - compZMovingSITK, compXMovingCupy1 - compXMovingSITK, alpha=0.5, color='green', angles='xy', scale_units='xy', scale=2, width=.010)
        ax[1, 2].set_xlabel('Field diff cupy1-sitk')
        ax[1, 3].quiver(compZMovingCupy1 - compZMovingCupy3, compXMovingCupy1 - compXMovingCupy3, alpha=0.5, color='green', angles='xy', scale_units='xy', scale=2, width=.010)
        ax[1, 3].set_xlabel('Field diff cupy1-cupy3')
    if showMask:
        ax[0, 0].imshow(maskFixed.getBinaryContourMask(internalBorder=True).imageArray[:, y_slice, :], alpha=0.5, cmap='Reds')
        ax[0, 1].imshow(maskMovingCupy1.getBinaryContourMask(internalBorder=True).imageArray[:, y_slice, :], alpha=0.5, cmap='Reds')
        ax[0, 2].imshow(maskMovingCupy3.getBinaryContourMask(internalBorder=True).imageArray[:, y_slice, :], alpha=0.5, cmap='Reds')
        ax[0, 3].imshow(maskMovingSitk.getBinaryContourMask(internalBorder=True).imageArray[:, y_slice, :], alpha=0.5, cmap='Reds')
        ax[2, 0].imshow(maskMovingCupy1.getBinaryContourMask(internalBorder=True).imageArray[:, y_slice, :] ^ maskMovingSitk.getBinaryContourMask(internalBorder=True).imageArray[:, y_slice, :], alpha=0.5, cmap='Reds')
        ax[2, 0].set_xlabel('Mask diff cupy1-sitk')
        ax[2, 1].imshow(maskMovingCupy1.getBinaryContourMask(internalBorder=True).imageArray[:, y_slice,:] ^ maskMovingCupy3.getBinaryContourMask(internalBorder=True).imageArray[:, y_slice, :], alpha=0.5, cmap='Reds')
        ax[2, 1].set_xlabel('Mask diff cupy1-cupy3')

    ax[0, 0].set_title('Fixed')
    ax[0, 0].set_xlabel(f"{fixed.origin}\n{fixed.spacing}\n{fixed.gridSize}")
    ax[0, 1].set_title('Moving Cupy1')
    ax[0, 1].set_xlabel(f"{movingCupy1.origin}\n{movingCupy1.spacing}\n{movingCupy1.gridSize}")
    ax[0, 2].set_title('Moving Cupy3')
    ax[0, 2].set_xlabel(f"{movingCupy3.origin}\n{movingCupy3.spacing}\n{movingCupy3.gridSize}")
    ax[0, 3].set_title('Moving SITK')
    ax[0, 3].set_xlabel(f"{movingSitk.origin}\n{movingSitk.spacing}\n{movingSitk.gridSize}")

    plt.savefig(os.path.join(output_path, 'cupy_VS_sitk_Transforms.png'))
    plt.show()

#%%
# Test using a Transform3D
# -----------------------------
print('-' * 40)

movingCupy1 = copy.deepcopy(fixed)
movingCupy3 = copy.deepcopy(fixed)
movingSitk = copy.deepcopy(fixed)
fieldMovingCupy1 = copy.deepcopy(fieldFixed)
fieldMovingCupy3 = copy.deepcopy(fieldFixed)
fieldMovingSitk = copy.deepcopy(fieldFixed)
maskMovingCupy1 = copy.deepcopy(maskFixed)
maskMovingCupy3 = copy.deepcopy(maskFixed)
maskMovingSitk = copy.deepcopy(maskFixed)

## Create a transform 3D
print('Create a transform 3D')
transform3D = Transform3D()
transform3D.initFromTranslationAndRotationVectors(transVec=translation, rotVec=rotation)
transform3D.setCenter(rotCenter)
print('Translation', transform3D.getTranslation())
print('Rotation', transform3D.getRotationAngles(inDegrees=True))

print('Moving with transform3D')
movingCupy1 = transform3D.deformData(movingCupy1, outputBox=outputBox, fillValue=-1000, tryGPU=True, interpOrder=1)
fieldMovingCupy1 = transform3D.deformData(fieldMovingCupy1, outputBox=outputBox, tryGPU=True, interpOrder=1)
maskMovingCupy1 = transform3D.deformData(maskMovingCupy1, outputBox=outputBox, tryGPU=True, interpOrder=1)

movingCupy3 = transform3D.deformData(movingCupy3, outputBox=outputBox, fillValue=-1000, tryGPU=True, interpOrder=3)
fieldMovingCupy3 = transform3D.deformData(fieldMovingCupy3, outputBox=outputBox, tryGPU=True, interpOrder=3)
maskMovingCupy3 = transform3D.deformData(maskMovingCupy3, outputBox=outputBox, tryGPU=True, interpOrder=3)

movingSitk = transform3D.deformData(movingSitk, outputBox=outputBox, fillValue=-1000)
fieldMovingSitk = transform3D.deformData(fieldMovingSitk, outputBox=outputBox)
maskMovingSitk = transform3D.deformData(maskMovingSitk, outputBox=outputBox)

showImagesAndFieldAndMask(fixed, movingCupy1, movingCupy3, movingSitk, fieldFixed, fieldMovingCupy1,
                            fieldMovingCupy3, fieldMovingSitk,
                            maskFixed, maskMovingCupy1, maskMovingCupy3, maskMovingSitk, y_slice,
                            figTitle='Test using a Transform3D', showField=showField, showImage=showImage,
                            showMask=showMask)

#%%
# Test using translateData
# -----------------------------
print('-' * 40)

movingCupy1 = copy.deepcopy(fixed)
movingCupy3 = copy.deepcopy(fixed)
movingSitk = copy.deepcopy(fixed)
fieldMovingCupy1 = copy.deepcopy(fieldFixed)
fieldMovingCupy3 = copy.deepcopy(fieldFixed)
fieldMovingSitk = copy.deepcopy(fieldFixed)
maskMovingCupy1 = copy.deepcopy(maskFixed)
maskMovingCupy3 = copy.deepcopy(maskFixed)
maskMovingSitk = copy.deepcopy(maskFixed)

print('Moving with translateData')
translateData(movingCupy1, translationInMM=translation, outputBox=outputBox, fillValue=-1000, tryGPU=True, interpOrder=1)
translateData(fieldMovingCupy1, translationInMM=translation, outputBox=outputBox, tryGPU=True, interpOrder=1)
translateData(maskMovingCupy1, translationInMM=translation, outputBox=outputBox, tryGPU=True, interpOrder=1)

translateData(movingCupy3, translationInMM=translation, outputBox=outputBox, fillValue=-1000, tryGPU=True, interpOrder=3)
translateData(fieldMovingCupy3, translationInMM=translation, outputBox=outputBox, tryGPU=True, interpOrder=3)
translateData(maskMovingCupy3, translationInMM=translation, outputBox=outputBox, tryGPU=True, interpOrder=3)

translateData(movingSitk, translationInMM=translation, outputBox=outputBox, fillValue=-1000)
translateData(fieldMovingSitk, translationInMM=translation, outputBox=outputBox)
translateData(maskMovingSitk, translationInMM=translation, outputBox=outputBox)

showImagesAndFieldAndMask(fixed, movingCupy1, movingCupy3, movingSitk, fieldFixed, fieldMovingCupy1,
                            fieldMovingCupy3, fieldMovingSitk,
                            maskFixed, maskMovingCupy1, maskMovingCupy3, maskMovingSitk, y_slice,
                            figTitle='Test using translateData', showField=showField, showImage=showImage,
                            showMask=showMask)

#%%
# Test using rotateData
# -----------------------------
print('-' * 40)

movingCupy1 = copy.deepcopy(fixed)
movingCupy3 = copy.deepcopy(fixed)
movingSitk = copy.deepcopy(fixed)
fieldMovingCupy1 = copy.deepcopy(fieldFixed)
fieldMovingCupy3 = copy.deepcopy(fieldFixed)
fieldMovingSitk = copy.deepcopy(fieldFixed)
maskMovingCupy1 = copy.deepcopy(maskFixed)
maskMovingCupy3 = copy.deepcopy(maskFixed)
maskMovingSitk = copy.deepcopy(maskFixed)

print('Moving with rotateData')
rotateData(movingCupy1, rotAnglesInDeg=rotation, outputBox=outputBox, fillValue=-1000, rotCenter=rotCenter, tryGPU=True, interpOrder=1)
rotateData(fieldMovingCupy1, rotAnglesInDeg=rotation, outputBox=outputBox, rotCenter=rotCenter, tryGPU=True, interpOrder=1)
rotateData(maskMovingCupy1, rotAnglesInDeg=rotation, outputBox=outputBox, rotCenter=rotCenter, tryGPU=True, interpOrder=1)

rotateData(movingCupy3, rotAnglesInDeg=rotation, outputBox=outputBox, fillValue=-1000, rotCenter=rotCenter, tryGPU=True, interpOrder=3)
rotateData(fieldMovingCupy3, rotAnglesInDeg=rotation, outputBox=outputBox, rotCenter=rotCenter, tryGPU=True, interpOrder=3)
rotateData(maskMovingCupy3, rotAnglesInDeg=rotation, outputBox=outputBox, rotCenter=rotCenter, tryGPU=True, interpOrder=3)

rotateData(movingSitk, rotAnglesInDeg=rotation, outputBox=outputBox, fillValue=-1000, rotCenter=rotCenter)
rotateData(fieldMovingSitk, rotAnglesInDeg=rotation, outputBox=outputBox, rotCenter=rotCenter)
rotateData(maskMovingSitk, rotAnglesInDeg=rotation, outputBox=outputBox, rotCenter=rotCenter)

showImagesAndFieldAndMask(fixed, movingCupy1, movingCupy3, movingSitk, fieldFixed, fieldMovingCupy1,
                            fieldMovingCupy3, fieldMovingSitk,
                            maskFixed, maskMovingCupy1, maskMovingCupy3, maskMovingSitk, y_slice,
                            figTitle='Test using rotateData', showField=showField, showImage=showImage,
                            showMask=showMask)

