'''
Simple IMPT plan optimization
=============================
In this example, we will create and optimize a simple Protons plan.
'''

#imports
import math
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

#%%
#import the needed opentps.core packages

from opentps.core.data.images import CTImage
from opentps.core.data.images import ROIMask
from opentps.core.data.plan import ObjectivesList
from opentps.core.data.plan import ProtonPlanDesign
from opentps.core.data import DVH
from opentps.core.data import Patient
from opentps.core.data.plan import FidObjective
from opentps.core.io import mcsquareIO
from opentps.core.io.scannerReader import readScanner
from opentps.core.io.serializedObjectIO import saveRTPlan, loadRTPlan
from opentps.core.processing.doseCalculation.doseCalculationConfig import DoseCalculationConfig
from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import MCsquareDoseCalculator
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D, resampleImage3D
from opentps.core.processing.planOptimization.planOptimization import IntensityModulationOptimizer

#%%
#CT calibration and BDL
#----------------------

ctCalibration = readScanner(DoseCalculationConfig().scannerFolder)
bdl = mcsquareIO.readBDL(DoseCalculationConfig().bdlFile)

#%%
#Create synthetic CT and ROI
#---------------------------

patient = Patient()
patient.name = 'Patient'

ctSize = 150

ct = CTImage()
ct.name = 'CT'
ct.patient = patient


huAir = -1024.
huWater = ctCalibration.convertRSP2HU(1.)
data = huAir * np.ones((ctSize, ctSize, ctSize))
data[:, 50:, :] = huWater
ct.imageArray = data

roi = ROIMask()
roi.patient = patient
roi.name = 'TV'
roi.color = (255, 0, 0) # red
data = np.zeros((ctSize, ctSize, ctSize)).astype(bool)
data[100:120, 100:120, 100:120] = True
roi.imageArray = data

#%%
#Configure dose engine
#---------------------

mc2 = MCsquareDoseCalculator()
mc2.beamModel = bdl
mc2.nbPrimaries = 5e4
mc2.ctCalibration = ctCalibration

mc2._independentScoringGrid = True
scoringSpacing = [2, 2, 2]
mc2._scoringVoxelSpacing = scoringSpacing

#%%
#Design plan
#-----------

beamNames = ["Beam1"]
gantryAngles = [0.]
couchAngles = [0.]

planInit = ProtonPlanDesign()
planInit.ct = ct
planInit.gantryAngles = gantryAngles
planInit.beamNames = beamNames
planInit.couchAngles = couchAngles
planInit.calibration = ctCalibration
planInit.spotSpacing = 6.0
planInit.layerSpacing = 6.0
planInit.targetMargin = 0.0
planInit.setScoringParameters(scoringSpacing=[2, 2, 2], adapt_gridSize_to_new_spacing=True)
# needs to be called after scoringGrid settings but prior to spot placement
planInit.defineTargetMaskAndPrescription(target = roi, targetPrescription = 20.) 
        
plan = planInit.buildPlan()  # Spot placement
plan.PlanName = "NewPlan"

beamlets = mc2.computeBeamlets(ct, plan, roi=[roi])
plan.planDesign.beamlets = beamlets
# doseImageRef = beamlets.toDoseImage()

#%%
#objectives
#----------

plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.DMAX, 20.0, 1.0)
plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.DMIN, 20.5, 1.0)
# Other examples of objectives
# plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.DMEAN, 20, 1.0) 
# plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.DUNIFORM, 20, 1.0)
# plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.DVHMIN, 19, 1.0, volume = 95)
# plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.DVHMAX, 21, 1.0, volume = 5)
# plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.EUDMIN, 19.5, 1.0, EUDa = 0.2)
# plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.EUDMAX, 20, 1.0, EUDa = 1)
# plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.EUDUNIFORM, 20.5, 1.0, EUDa = 0.5)
# plan.planDesign.objectives.addFidObjective(BODY, FidObjective.Metrics.DFALLOFF, weight=10, fallOffDistance=1, fallOffLowDoseLevel=0, fallOffHighDoseLevel=21)

#%%
#Optimize plan
#-------------

solver = IntensityModulationOptimizer(method='Scipy_L-BFGS-B', plan=plan, maxiter=50)
doseImage, ps = solver.optimize()

#%%
#Final dose computation
#----------------------

mc2.nbPrimaries = 1e7
doseImage = mc2.computeDose(ct, plan)

#%%
#Plots
#-----

# Compute DVH on resampled contour
roiResampled = resampleImage3D(roi, origin=ct.origin, spacing=scoringSpacing)
target_DVH = DVH(roiResampled, doseImage)
print('D95 = ' + str(target_DVH.D95) + ' Gy')
print('D5 = ' + str(target_DVH.D5) + ' Gy')
print('D5 - D95 =  {} Gy'.format(target_DVH.D5 - target_DVH.D95))

# center of mass
roi = resampleImage3DOnImage3D(roi, ct)
COM_coord = roi.centerOfMass
COM_index = roi.getVoxelIndexFromPosition(COM_coord)
Z_coord = COM_index[2]

img_ct = ct.imageArray[:, :, Z_coord].transpose(1, 0)
contourTargetMask = roi.getBinaryContourMask()
img_mask = contourTargetMask.imageArray[:, :, Z_coord].transpose(1, 0)
img_dose = resampleImage3DOnImage3D(doseImage, ct)
img_dose = img_dose.imageArray[:, :, Z_coord].transpose(1, 0)

#Output path
output_path = 'Output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Display dose
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(img_ct, cmap='gray')
ax[0].imshow(img_mask, alpha=.2, cmap='binary')  # PTV
dose = ax[0].imshow(img_dose, cmap='jet', alpha=.2)
plt.colorbar(dose, ax=ax[0])
ax[1].plot(target_DVH.histogram[0], target_DVH.histogram[1], label=target_DVH.name)
ax[1].set_xlabel("Dose (Gy)")
ax[1].set_ylabel("Volume (%)")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path, 'SimpleOpti1.png'),format = 'png')
plt.show()