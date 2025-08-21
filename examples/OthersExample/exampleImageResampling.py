'''
Image Resampling 
=========================
author: OpenTPS team

This example shows how the resampling function can be used on image3D objects.

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
import sys
import matplotlib.pyplot as plt
import os
from pathlib import Path

#%%
#import the needed opentps.core packages
from opentps.core.io.dataLoader import readData
from opentps.core.processing.imageProcessing.resampler3D import resample, resampleOnImage3D

#%%
# Get the current working directory, its parent, then add the testData folder at the end of it
testDataPath = os.path.join(Path(os.getcwd()).parent.absolute(), 'opentps/testData/')

#%%
#load an image to use as example
dataPath = testDataPath + "4DCTDicomLight/00"
img = readData(dataPath)[0]
print('Image type =', type(img))
zSlice = int(img.gridSize[2] / 2)

#%%
# let's resample the image using a specific spacing (upsampling or downsampling)
spacingResampledDown = resample(img, img.spacing * 1.5)
spacingResampledDownZSlice = int(spacingResampledDown.gridSize[2] / 2)
"""
Note that as the spacing is the second argument in the resample function, it can be use without specifying the argument name if put in the second position (here above)
If you prefer to be sure to specify the correct argument, you can use the name as the example here under
"""
spacingResampledUp = resample(img, spacing=img.spacing * 0.5)
spacingResampledUpZSlice = int(spacingResampledUp.gridSize[2] / 2)

#%%
# display results with spacing, gridSize and origin
fig, ax = plt.subplots(1, 3)

ax[0].imshow(img.imageArray[:, :, zSlice])
ax[0].set_title(f"Original image")
ax[0].set_xlabel(f"Spacing, {img.spacing} \n Grid Size {img.gridSize} \n Origin {img.origin}")

ax[1].imshow(spacingResampledDown.imageArray[:, :, spacingResampledDownZSlice])
ax[1].set_title(f"Downsampled using spacing")
ax[1].set_xlabel(f"Spacing, {spacingResampledDown.spacing} \n Grid Size {spacingResampledDown.gridSize} \n Origin {spacingResampledDown.origin}")

ax[2].imshow(spacingResampledUp.imageArray[:, :, spacingResampledUpZSlice])
ax[2].set_title(f"Upsampled using spacing")
ax[2].set_xlabel(f"Spacing, {spacingResampledUp.spacing} \n Grid Size {spacingResampledUp.gridSize} \n Origin {spacingResampledUp.origin}")

plt.show()

#%%
# now let's resample the image using a specific gridSize
gridSizeResampled = resample(img, gridSize=(200, 200, 200))
gridSizeResampledZSlice = int(gridSizeResampled.gridSize[2] / 2)

#%%
# and using BOTH a specific spacing AND gridSize
"""
Note that this is not recomanded as it can push parts of the image outside the new array grid and have the same effect as cropping the data
"""
spacingAndGSResampled = resample(img, gridSize=(100, 100, 100), spacing=(2, 2, 2))
spacingAndGSResampledZSlice = int(spacingAndGSResampled.gridSize[2] / 2)

#%%
# display results with spacing, gridSize and origin
fig, ax = plt.subplots(1, 3)

ax[0].imshow(img.imageArray[:, :, zSlice])
ax[0].set_title(f"Original image")
ax[0].set_xlabel(f"Spacing, {img.spacing} \n Grid Size {img.gridSize} \n Origin {img.origin}")

ax[1].imshow(gridSizeResampled.imageArray[:, :, gridSizeResampledZSlice])
ax[1].set_title(f"Resampled using gridSize")
ax[1].set_xlabel(f"Spacing, {gridSizeResampled.spacing} \n Grid Size {gridSizeResampled.gridSize} \n Origin {gridSizeResampled.origin}")

ax[2].imshow(spacingAndGSResampled.imageArray[:, :, spacingAndGSResampledZSlice])
ax[2].set_title(f"Resampled using gridSize AND spacing")
ax[2].set_xlabel(f"Spacing, {spacingAndGSResampled.spacing} \n Grid Size {spacingAndGSResampled.gridSize} \n Origin {spacingAndGSResampled.origin}")

plt.show()

#%%
# now let's try using the origin, which corresponds to a translation of the image
originResampled = resample(img, origin=(-220, -200, -200))
originResampledZSlice = int(originResampled.gridSize[2] / 2)

#%%
# and using BOTH the origin AND gridSize
"""
Note that this is not recomanded as it can push parts of the image outside the new array grid and have the same effect as cropping the data
"""
originAndGSResampled = resample(img, gridSize=(50, 50, 50), origin=(-220, -200, -200))
originAndGSResampledZSlice = int(originAndGSResampled.gridSize[2] / 2)

#%%
# display results with spacing, gridSize and origin
fig, ax = plt.subplots(1, 3)

ax[0].imshow(img.imageArray[:, :, zSlice])
ax[0].set_title(f"Original image")
ax[0].set_xlabel(f"Spacing, {img.spacing} \n Grid Size {img.gridSize} \n Origin {img.origin}")

ax[1].imshow(originResampled.imageArray[:, :, originResampledZSlice])
ax[1].set_title(f"Resampled using origin")
ax[1].set_xlabel(f"Spacing, {originResampled.spacing} \n Grid Size {originResampled.gridSize} \n Origin {originResampled.origin}")

ax[2].imshow(originAndGSResampled.imageArray[:, :, originAndGSResampledZSlice])
ax[2].set_title(f"Resampled using gridSize AND origin")
ax[2].set_xlabel(f"Spacing, {originAndGSResampled.spacing} \n Grid Size {originAndGSResampled.gridSize} \n Origin {originAndGSResampled.origin}")

plt.show()

#%%
# Now you can also use the following function if you need to resample an image on the grid of another image
resampledOnGrid = resampleOnImage3D(gridSizeResampled, fixedImage=img)

#%%
# where the spacing, gridSize and origin of the fixedImage is used to resample the data (first argument)

resampledOnGridZSlice = int(resampledOnGrid.gridSize[2] / 2)

fig, ax = plt.subplots(1, 2)

ax[0].imshow(gridSizeResampled.imageArray[:, :, gridSizeResampledZSlice])
ax[0].set_title(f"Before resampling")
ax[0].set_xlabel(f"Spacing, {gridSizeResampled.spacing} \n Grid Size {gridSizeResampled.gridSize} \n Origin {gridSizeResampled.origin}")

ax[1].imshow(resampledOnGrid.imageArray[:, :, resampledOnGridZSlice])
ax[1].set_title(f"After resampling")
ax[1].set_xlabel(f"Spacing, {resampledOnGrid.spacing} \n Grid Size {resampledOnGrid.gridSize} \n Origin {resampledOnGrid.origin}")

plt.show()
