'''
Simple ion plan optimization and DICOM study creation
=====================================================
author: OpenTPS team

In this example, we will create and optimize a simple ion (Proton) plan. 
The generated CT, the plan, and the dose will be saved as DICOM files.

running time: ~ 15 minutes
'''

#%% 
# Setting up the environment in google collab
#--------------
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
import logging
import numpy as np
from matplotlib import pyplot as plt
import sys
import datetime
import pydicom
sys.path.append('..')

#%%
#import the needed opentps.core packages

from opentps.core.io.dicomIO import writeRTPlan, writeDicomCT, writeRTDose, writeRTStruct
from opentps.core.processing.planOptimization.tools import evaluateClinical
from opentps.core.data.images import CTImage, DoseImage
from opentps.core.data.images import ROIMask
from opentps.core.data.plan import ObjectivesList
from opentps.core.data.plan._protonPlanDesign import ProtonPlanDesign
from opentps.core.data import DVH
from opentps.core.data import Patient
from opentps.core.data import RTStruct
from opentps.core.data.plan import FidObjective
from opentps.core.io import mcsquareIO
from opentps.core.io.scannerReader import readScanner
from opentps.core.io.serializedObjectIO import saveRTPlan, loadRTPlan
from opentps.core.processing.doseCalculation.doseCalculationConfig import DoseCalculationConfig
from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import MCsquareDoseCalculator
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D, resampleImage3D
from opentps.core.processing.planOptimization.planOptimization import IntensityModulationOptimizer
from opentps.core.data.plan import ProtonPlan

logger = logging.getLogger(__name__)

#%%
#CT calibration and BDL
#----------------------

ctCalibration = readScanner(DoseCalculationConfig().scannerFolder)
bdl = mcsquareIO.readBDL(DoseCalculationConfig().bdlFile)

#%%
#Create the DICOM files
#----------------------

 # ++++Don't delete UIDs to build the simple study+++++++++++++++++++
studyInstanceUID = pydicom.uid.generate_uid()
doseSeriesInstanceUID = pydicom.uid.generate_uid()
planSeriesInstanceUID = pydicom.uid.generate_uid()
ctSeriesInstanceUID =  pydicom.uid.generate_uid()
frameOfReferenceUID = pydicom.uid.generate_uid()
# structSeriesInstanceUID = pydicom.uid.generate_uid()
dt = datetime.datetime.now()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%
#Output path
#-----------

output_path = 'Output'
if not os.path.exists(output_path):
    os.makedirs(output_path)


#%%
#Generic CT creation
#-------------------
#we will first create a generic CT of a box fill with water and air

patient = Patient()
patient.name = 'Simple_Patient'
Patient.id = 'Simple_Patient'
Patient.birthDate = dt.strftime('%Y%m%d')
patient.sex = ""

ctSize = 150
ct = CTImage(seriesInstanceUID=ctSeriesInstanceUID, frameOfReferenceUID=frameOfReferenceUID)
ct.name = 'CT'
ct.patient = patient
ct.studyInstanceUID = studyInstanceUID

huAir = -1024.
huWater = ctCalibration.convertRSP2HU(1.)
data = huAir * np.ones((ctSize, ctSize, ctSize))
data[:, 50:, :] = huWater
ct.imageArray = data
writeDicomCT(ct, output_path)

#%%
#Region of interest
#------------------
#we will now create a region of interest wich is a small 3D box of size 20*20*20

roi = ROIMask()
roi.patient = patient
roi.name = 'TV'
roi.color = (255, 0, 0)  # red
data = np.zeros((ctSize, ctSize, ctSize)).astype(bool)
data[100:120, 100:120, 100:120] = True
roi.imageArray = data

#%%
#Configuration of Mcsquare
#-------------------------
#To configure the MCsquare calculator we need to calibrate it with the CT calibration obtained above

mc2 = MCsquareDoseCalculator()
mc2.beamModel = bdl
mc2.nbPrimaries = 5e4
mc2.ctCalibration = ctCalibration

#%%
# Plan Creation
#--------------

 # Design plan
beamNames = ["Beam1"]
gantryAngles = [0.]
couchAngles = [0.]

# Load / Generate new plan
plan_file = os.path.join(output_path, "Plan_WaterPhantom_cropped_resampled.tps")

if os.path.isfile(plan_file):
    plan = loadRTPlan(plan_file)
    logger.info('Plan loaded')
else:
    planDesign = ProtonPlanDesign()
    planDesign.ct = ct
    planDesign.targetMask = roi
    planDesign.gantryAngles = gantryAngles
    planDesign.beamNames = beamNames
    planDesign.couchAngles = couchAngles
    planDesign.calibration = ctCalibration
    planDesign.spotSpacing = 5.0
    planDesign.layerSpacing = 5.0
    planDesign.targetMargin = 5.0
    planDesign.setScoringParameters(scoringSpacing=[2, 2, 2], adapt_gridSize_to_new_spacing=True)
    planDesign.defineTargetMaskAndPrescription(target = roi, targetPrescription = 20.) # needs to be called prior spot placement
    
    plan = planDesign.buildPlan()  # Spot placement
    plan.rtPlanName = "Simple_Patient"

    beamlets = mc2.computeBeamlets(ct, plan, roi=[roi])
    plan.planDesign.beamlets = beamlets
    beamlets.storeOnFS(os.path.join(output_path, "BeamletMatrix_" + plan.seriesInstanceUID + ".blm"))
    # Save plan with initial spot weights in serialized format (OpenTPS format)
    saveRTPlan(plan, plan_file)
    writeRTPlan(plan, output_path)

#%%
#objectives
#----------

# Set objectives (attribut is already initialized in planDesign object)
plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.DMAX, 20.0, 1.0)
plan.planDesign.objectives.addFidObjective(roi, FidObjective.Metrics.DMIN, 20.5, 1.0)

#%%
#Optimize plan
#-------------

plan.seriesInstanceUID = planSeriesInstanceUID
plan.studyInstanceUID = studyInstanceUID
plan.frameOfReferenceUID = frameOfReferenceUID
plan.rtPlanGeometry = "TREATMENT_DEVICE"

solver = IntensityModulationOptimizer(method='Scipy_L-BFGS-B', plan=plan, maxiter=1000)
# Optimize treatment plan
doseImage, ps = solver.optimize()

# Save plan with updated spot weights in serialized format (OpenTPS format)
plan_file_optimized = os.path.join(output_path, "Plan_WaterPhantom_cropped_resampled_optimized.tps")
saveRTPlan(plan, plan_file_optimized)
# Save plan with updated spot weights in dicom format
plan.patient = patient
writeRTPlan(plan, output_path)

#%%
#Dose volume histogram
#---------------------

target_DVH = DVH(roi, doseImage)
print('D5 - D95 =  {} Gy'.format(target_DVH.D5 - target_DVH.D95))
clinROI = [roi.name, roi.name]
clinMetric = ["Dmin", "Dmax"]
clinLimit = [19., 21.]
clinObj = {'ROI': clinROI, 'Metric': clinMetric, 'Limit': clinLimit}
print('Clinical evaluation')
evaluateClinical(doseImage, [roi], clinObj)

doseImage.referencePlan = plan
doseImage.referenceCT = ct
doseImage.patient = patient
doseImage.studyInstanceUID = studyInstanceUID
doseImage.frameOfReferenceUID = frameOfReferenceUID 
doseImage.sopClassUID = '1.2.840.10008.5.1.4.1.1.481.2'
doseImage.mediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'
doseImage.sopInstanceUID = pydicom.uid.generate_uid()
doseImage.studyTime = dt.strftime('%H%M%S.%f')
doseImage.studyDate = dt.strftime('%Y%m%d')
doseImage.SOPInstanceUID = doseImage.sopInstanceUID

if not hasattr(ProtonPlan, "SOPInstanceUID"):
    ProtonPlan.SOPInstanceUID = property(lambda self: self.sopInstanceUID)


writeRTDose(doseImage, output_path)

#%%
#Center of mass
#--------------
#Here we look at the part of the 3D CT image where "stuff is happening" by getting the CoM. We use the function resampleImage3DOnImage3D to the same array size for both images.

roi = resampleImage3DOnImage3D(roi, ct)
COM_coord = roi.centerOfMass
COM_index = roi.getVoxelIndexFromPosition(COM_coord)
Z_coord = COM_index[2]

img_ct = ct.imageArray[:, :, Z_coord].transpose(1, 0)
contourTargetMask = roi.getBinaryContourMask()
img_mask = contourTargetMask.imageArray[:, :, Z_coord].transpose(1, 0)
img_dose = resampleImage3DOnImage3D(doseImage, ct)
img_dose = img_dose.imageArray[:, :, Z_coord].transpose(1, 0)

#%%
#Plot of the dose
#----------------

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].axes.get_xaxis().set_visible(False)
ax[0].axes.get_yaxis().set_visible(False)
ax[0].imshow(img_ct, cmap='gray')
ax[0].imshow(img_mask, alpha=.2, cmap='binary')  # PTV
dose = ax[0].imshow(img_dose, cmap='jet', alpha=.2)
plt.colorbar(dose, ax=ax[0])
ax[1].plot(target_DVH.histogram[0], target_DVH.histogram[1], label=target_DVH.name)
ax[1].set_xlabel("Dose (Gy)")
ax[1].set_ylabel("Volume (%)")
ax[1].grid(True)
ax[1].legend()

convData = solver.getConvergenceData()
ax[2].plot(np.linspace(0, convData['time'], len(convData['func_0'])), convData['func_0'], 'bo-', lw=2,
            label='Fidelity')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Cost')
ax[2].set_yscale('symlog')
ax2 = ax[2].twiny()
ax2.set_xlabel('Iterations')
ax2.set_xlim(0, convData['nIter'])
ax[2].grid(True)

plt.savefig(f'{output_path}/Dose_SimpleOptimization.png', format = 'png')
plt.show()