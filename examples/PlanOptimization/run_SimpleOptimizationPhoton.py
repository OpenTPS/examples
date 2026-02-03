'''
Simple IMRT photon plan optimization
====================================
author: OpenTPS team

In this example, we will create and optimize a simple Photons plan.

running time: ~ 15 minutes
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

import os
import logging
import numpy as np
from matplotlib import pyplot as plt
import sys
import copy
from scipy.sparse import csc_matrix
sys.path.append('..')


#%%
#import the needed opentps.core packages

from opentps.core.io.dicomIO import writeRTDose, writeDicomCT, writeRTPlan, writeRTStruct
from opentps.core.processing.planOptimization.tools import evaluateClinical
from opentps.core.data.images import CTImage
from opentps.core.data.images import ROIMask
from opentps.core.data import DVH
from opentps.core.data import Patient
from opentps.core.io.scannerReader import readScanner
from opentps.core.io.serializedObjectIO import saveRTPlan, loadRTPlan
from opentps.core.processing.doseCalculation.doseCalculationConfig import DoseCalculationConfig
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
from opentps.core.processing.planOptimization.planOptimization import  IntensityModulationOptimizer
from opentps.core.processing.doseCalculation.photons.cccDoseCalculator import CCCDoseCalculator
from opentps.core.data.plan import PhotonPlanDesign
import opentps.core.processing.planOptimization.objectives.dosimetricObjectives as doseObj

logger = logging.getLogger(__name__)

#%%
#CT calibration
#--------------

ctCalibration = readScanner(DoseCalculationConfig().scannerFolder)

#%%
#Create synthetic CT and ROI
#---------------------------

patient = Patient()
patient.name = 'Simple_Patient'

ctSize = 150
ct = CTImage()
ct.name = 'CT'
ct.patient = patient
# ct.origin = -ctSize/2 * ct.spacing

huAir = -1024.
huWater = 0
data = huAir * np.ones((ctSize, ctSize, ctSize))
data[:, 50:, :] = huWater
ct.imageArray = data
#writeDicomCT(ct, output_path)

# Struct
roi = ROIMask()
roi.patient = patient
# roi.origin = -ctSize/2 * ct.spacing
roi.name = 'TV'
roi.color = (255, 0, 0)  # red
data = np.zeros((ctSize, ctSize, ctSize)).astype(bool)
data[100:120, 100:120, 100:120] = True
roi.imageArray = data

body = roi.copy()
body.name = 'Body'
body.dilateMask(20)
body.imageArray = np.logical_xor(body.imageArray, roi.imageArray).astype(bool)


#%%
#Output path
#-----------

output_path = 'Output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

#%%
# Plan Creation
#--------------

# Design plan
beamNames = ["Beam1", "Beam2"]
gantryAngles = [0., 90.]
couchAngles = [0.,0]

## Dose computation from plan
ccc = CCCDoseCalculator(batchSize= 30)
ccc.ctCalibration = readScanner(DoseCalculationConfig().scannerFolder)

# Load / Generate new plan
plan_file = os.path.join(output_path, "PhotonPlan_WaterPhantom_cropped_resampled.tps")

if os.path.isfile(plan_file): ### Remove the False to load the plan
    plan = loadRTPlan(plan_file, radiationType='photon')
    logger.info('Plan loaded')
else:
    planDesign = PhotonPlanDesign()
    planDesign.ct = ct
    planDesign.targetMask = roi
    planDesign.isocenterPosition_mm = None # None take the center of mass of the target
    planDesign.gantryAngles = gantryAngles
    planDesign.couchAngles = couchAngles
    planDesign.beamNames = beamNames
    planDesign.calibration = ctCalibration
    planDesign.xBeamletSpacing_mm = 5
    planDesign.yBeamletSpacing_mm = 5
    planDesign.targetMargin = 5.0
    planDesign.defineTargetMaskAndPrescription(target = roi, targetPrescription = 20.)

    plan = planDesign.buildPlan()

    beamlets = ccc.computeBeamlets(ct, plan)
    doseInfluenceMatrix = copy.deepcopy(beamlets)

    plan.planDesign.beamlets = beamlets
    beamlets.storeOnFS(os.path.join(output_path, "BeamletMatrix_" + plan.seriesInstanceUID + ".blm"))
    # Save plan with initial spot weights in serialized format (OpenTPS format)
    saveRTPlan(plan, plan_file)

plan.planDesign.objectives.addObjective(doseObj.DMax(body,5, weight=1.0))

plan.planDesign.objectives.addObjective(doseObj.DMax(roi, 21, weight=10.0))
plan.planDesign.objectives.addObjective(doseObj.DMin(roi, 20, weight=20.0))

# Other examples of objectives

# plan.planDesign.objectives.addObjective(doseObj.DUniform(roi, 20, weight=10))
#
# plan.planDesign.objectives.addObjective(doseObj.DMaxMean(roi,21,weight=5))
# plan.planDesign.objectives.addObjective(doseObj.DMinMean(roi,20,weight=5))
# plan.planDesign.objectives.addObjective(doseObj.DFallOff(roi,oar,10,5,15,weight=5))
#
# plan.planDesign.objectives.addObjective(doseObj.DVHMax(roi, 21,0.05, weight=5))
# plan.planDesign.objectives.addObjective(doseObj.DVHMin(roi, 18,0.90, weight=5))
#
# plan.planDesign.objectives.addObjective(doseObj.EUDMin(roi, 19, 1, weight=50))
# plan.planDesign.objectives.addObjective(doseObj.EUDMax(roi, 21, 1, weight=50))
# plan.planDesign.objectives.addObjective(doseObj.EUDUniform(roi, 20, 1, weight=50))


plan.numberOfFractionsPlanned = 30

plan.planDesign.ROI_cropping = False # False, not cropping allows you to keep the dose outside the ROIs and then use the 'shift' evaluation method, which simply shifts the beamlets.
solver = IntensityModulationOptimizer(method='Scipy_L-BFGS-B', plan=plan, maxiter=1000)

#%%
# Optimize treatment plan
#------------------------
doseImage, ps = solver.optimize()
writeRTDose(doseImage, output_path)

# Save plan with updated spot weights in serialized format (OpenTPS format)
plan_file_optimized = os.path.join(output_path, "Plan_WaterPhantom_cropped_resampled_optimized.tps")
saveRTPlan(plan, plan_file_optimized)
# Save plan with updated spot weights in dicom format
plan.patient = patient
# writeRTPlan(plan, output_path, outputFilename = plan.name )
# writeDicomPhotonRTPlan(plan, output_path )
#%%
# Compute DVH on resampled contour
#---------------------------------

target_DVH = DVH(roi, doseImage)
body_DVH = DVH(body, doseImage)
print('D5 - D95 =  {} Gy'.format(target_DVH.D5 - target_DVH.D95))
clinROI = [roi.name, roi.name]
clinMetric = ["Dmin", "Dmax"]
clinLimit = [19., 21.]
clinObj = {'ROI': clinROI, 'Metric': clinMetric, 'Limit': clinLimit}
print('Clinical evaluation')
evaluateClinical(doseImage, [roi], clinObj)

#%%
# center of mass
#---------------
roi = resampleImage3DOnImage3D(roi, ct)
body = resampleImage3DOnImage3D(body,ct)
COM_coord = roi.centerOfMass
COM_index = roi.getVoxelIndexFromPosition(COM_coord)
Z_coord = COM_index[2]

img_ct = ct.imageArray[:, :, Z_coord].transpose(1, 0)
img_mask = roi.imageArray[:, :, Z_coord].transpose(1, 0)
img_body = body.imageArray[:, :, Z_coord].transpose(1, 0)
img_dose = resampleImage3DOnImage3D(doseImage, ct)
img_dose = img_dose.imageArray[:, :, Z_coord].transpose(1, 0)

#%%
# Display dose
#-------------
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].axes.get_xaxis().set_visible(False)
ax[0].axes.get_yaxis().set_visible(False)
ax[0].imshow(img_ct, cmap='gray')
ax[0].contour(img_body,[0.5],colors='green') # body
ax[0].contour(img_mask,[0.5],colors='red')  # PTV

dose = ax[0].imshow(img_dose, cmap='jet', alpha=.2)
plt.colorbar(dose, ax=ax[0])
ax[1].plot(target_DVH.histogram[0], target_DVH.histogram[1], label=target_DVH.name,color='red')
ax[1].plot(body_DVH.histogram[0], body_DVH.histogram[1], label=body_DVH.name,color='green')
ax[1].set_xlabel("Dose (Gy)")
ax[1].set_ylabel("Volume (%)")
ax[1].grid(True)
ax[1].legend()

convData = solver.getConvergenceData()
x_data = np.linspace(0, convData['time'], len(convData['func_0']))
y_data = convData['func_0']
ax[2].plot(x_data, y_data , 'bo-', lw=2, label='Fidelity')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Cost')
ax[2].set_yscale('symlog')
ax2 = ax[2].twiny()
ax2.set_xlabel('Iterations')
ax2.set_xlim(0, convData['nIter'])
ax[2].grid(True)
plt.savefig(os.path.join(output_path, 'Dose_SimpleOptimizationPhotons.png'))
plt.show()