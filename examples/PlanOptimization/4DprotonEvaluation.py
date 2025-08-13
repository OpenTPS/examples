'''
4D Proton Evaluation
=========================
author: OpenTPS team

This example shows how to evaluate a 4D proton plan using OpenTPS.
'''
# %% [skip]
# installing opentps in colab
import sys
if "google.colab" in sys.modules:
    from IPython import get_ipython
    get_ipython().system('git clone https://gitlab.com/openmcsquare/opentps.git')
    get_ipython().system('pip install ./opentps')
    get_ipython().system('!pip install cupy-cuda12x')
    import opentps

#%%
#imports
import os
import datetime
import logging
import pydicom
import datetime

import numpy as np
from matplotlib import pyplot as plt

#%%
# import the needed opentps.core packages
from opentps.core.data.images import CTImage
from opentps.core.data.images import ROIMask
from opentps.core.data.plan._protonPlanDesign import ProtonPlanDesign
from opentps.core.data import Patient
from opentps.core.io import mcsquareIO
from opentps.core.io.scannerReader import readScanner
from opentps.core.io.serializedObjectIO import saveRTPlan, loadRTPlan
from opentps.core.processing.doseCalculation.doseCalculationConfig import DoseCalculationConfig
from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import MCsquareDoseCalculator
from opentps.core.processing.planEvaluation.robustnessEvaluation import RobustnessEval
from opentps.core.io.dataLoader import readData
from opentps.core.data import DVH

logger = logging.getLogger(__name__)

#%%
# Output path
#------------

output_path = os.path.join(os.getcwd(), 'Exemple_Robust4DOptimization')
if not os.path.exists(output_path):
    os.makedirs(output_path)
logger.info('Files will be stored in {}'.format(output_path))

#%%
#CT calibration and BDL
#----------------------

ctCalibration = readScanner(DoseCalculationConfig().scannerFolder)
bdl = mcsquareIO.readBDL(DoseCalculationConfig().bdlFile)

#%%
# CT and ROI creation
#----------------------
# Generic example: 4DCT composed of 3 CTs : 2 phases and the MidP. 
# The anatomy consists of a square target moving vertically, with an organ at risk and soft tissue (muscle) in front of it. 
CT4D = []
ROI4D = []
for i in range(0, 3):
    # ++++Don't delete UIDs to build the simple study+++++++++++++++++++
    studyInstanceUID = pydicom.uid.generate_uid()
    ctSeriesInstanceUID =  pydicom.uid.generate_uid()
    frameOfReferenceUID = pydicom.uid.generate_uid()
    # structSeriesInstanceUID = pydicom.uid.generate_uid()
    dt = datetime.datetime.now()
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # CT
    patient = Patient()
    patient.name = f'Miro_OpenTPS_4DCT'
    Patient.id = f'12082024'
    Patient.birthDate = dt.strftime('%Y%m%d')
    patient.sex = ""
    
    ctSize = 150
    ct = CTImage(seriesInstanceUID=ctSeriesInstanceUID, frameOfReferenceUID=frameOfReferenceUID)
    ct.name = f'CT_Phase_{i}'
    ct.patient = patient
    ct.studyInstanceUID = studyInstanceUID

    huWater = 50
    huTarget = 100
    huMuscle = 200
    data = huWater * np.ones((ctSize, ctSize, ctSize))

    # Muscle
    data[100:140, 20:130, 55:95] = huMuscle
    # OAR
    data[70:80, 70:80, 65:85] = huTarget
    # TargetVolume
    if i == 0 :
        data[25:45, 70:100, 65:85] = huTarget
    if i == 1 :
        data[25:45, 60:90, 65:85] = huTarget
    if i == 2 :
        data[25:45, 50:80, 65:85] = huTarget
    ct.imageArray = data
    # writeDicomCT(ct, output_path)

    #---------------------ROI
    ROI = []

    # TargetVolume
    TV = ROIMask()
    TV.patient = patient
    TV.name = 'TV'
    TV.color = (255, 0, 0)  # red
    data = np.zeros((ctSize, ctSize, ctSize)).astype(bool)
    if i == 0 :
        data[25:45, 70:100, 65:85] = True
    if i == 1 :
        data[25:45, 60:90, 65:85] = True
    if i == 2 :
        data[25:45, 50:80, 65:85] = True
    TV.imageArray = data
    ROI.append(TV)
    
    # Muscle
    Muscle = ROIMask()
    Muscle.patient = patient
    Muscle.name = 'Muscle'
    Muscle.color = (150, 0, 0)
    data = np.zeros((ctSize, ctSize, ctSize)).astype(bool)
    data[100:140, 20:130, 55:95] = True
    Muscle.imageArray = data
    ROI.append(Muscle)

    # OAR
    OAR = ROIMask()
    OAR.patient = patient
    OAR.name = 'OAR'
    OAR.color = (100, 0, 0)
    data = np.zeros((ctSize, ctSize, ctSize)).astype(bool)
    data[70:80, 70:80, 65:85] = True
    OAR.imageArray = data
    ROI.append(OAR)

    # Body
    BODY = ROIMask()
    BODY.patient = patient
    BODY.name = 'Body'
    BODY.color = (100, 0, 0)
    data = np.ones((ctSize, ctSize, ctSize)).astype(bool)
    data[np.where(OAR.imageArray)] = False
    data[np.where(Muscle.imageArray)] = False
    data[np.where(TV.imageArray)] = False
    BODY.imageArray = data
    ROI.append(BODY)

    CT4D.append(ct)
    ROI4D.append(ROI)

RefCT = CT4D[1]
RefTV = ROI4D[1][0]

#%%
# Configure MCsquare
#----------------------
mc2 = MCsquareDoseCalculator()
mc2.beamModel = bdl
mc2.nbPrimaries = 5e4
mc2.statUncertainty = 2.
mc2.ctCalibration = ctCalibration

#%%
# Load / Generate new plan
#----------------------
plan_file = os.path.join(output_path, f"RobustPlan_4D_weighted.tps")

if os.path.isfile(plan_file):
    plan = loadRTPlan(plan_file)
    logger.info('Plan weighted loaded')
else:
    logger.error("You need to design and optimize a plan first - See 4DrobustOptimization script.")

#%%
# Load / Generate scenarios
#----------------------
scenario_folder = os.path.join(output_path, 'Robustness4D_Test')
if os.path.isdir(scenario_folder):
    scenarios = RobustnessEval()
    scenarios.selectionStrategy = RobustnessEval.Strategies.REDUCED_SET
    # scenarios.selectionStrategy = RobustnessEval.Strategies.ALL
    # scenarios.selectionStrategy = RobustnessEval.Strategies.RANDOM
    scenarios.setupSystematicError = plan.planDesign.robustnessEval.setupSystematicError
    scenarios.setupRandomError = plan.planDesign.robustnessEval.setupRandomError
    scenarios.rangeSystematicError = plan.planDesign.robustnessEval.rangeSystematicError
    scenarios.load(scenario_folder)
else:
    # MCsquare config for scenario dose computation
    mc2.nbPrimaries = 1e6
    plan.planDesign.robustnessEval = RobustnessEval()
    plan.planDesign.robustnessEval.setupSystematicError = [1.6, 1.6, 1.6]  # mm (sigma)
    plan.planDesign.robustnessEval.setupRandomError = [0.0, 0.0, 0.0]  # mm (sigma)
    plan.planDesign.robustnessEval.rangeSystematicError = 3.0  # %
    
    # 4D Evaluation mode
    plan.planDesign.robustnessEval.Mode4D = plan.planDesign.robustnessEval.Mode4D.MCsquareAccumulation # Or MCsquareSystematic

    # # 4D settings : only for the mode MCsquareAccumulation with the RANDOM strategie
    # plan.planDesign.robustnessEval.Create4DCTfromRef = True
    # plan.planDesign.robustnessEval.SystematicAmplitudeError = 5.0  # %
    # plan.planDesign.robustnessEval.RandomAmplitudeError = 5.0  # %
    # plan.planDesign.robustnessEval.Dynamic_delivery = True
    # plan.planDesign.robustnessEval.SystematicPeriodError = 5.0  # %  # Spot timing required. If not, we calculate them with SimpleBeamDeliveryTimings()
    # plan.planDesign.robustnessEval.RandomPeriodError = 5.0  # %
    # plan.planDesign.robustnessEval.Breathing_period = 1  # x100% 

    # Regular scenario sampling
    plan.planDesign.robustnessEval.selectionStrategy = plan.planDesign.robustnessEval.Strategies.REDUCED_SET

    # All scenarios (includes diagonals on sphere)
    # plan.planDesign.robustnessEval.selectionStrategy = plan.planDesign.robustnessEval.Strategies.ALL

    # Random scenario sampling  
    # plan.planDesign.robustnessEval.selectionStrategy = plan.planDesign.robustnessEval.Strategies.RANDOM
    # plan.planDesign.robustnessEval.numScenarios = 50 # Specify how many random scenarios to simulate, default = 100
    
    # Run MCsquare simulation
    scenarios = mc2.compute4DRobustScenario(CT4D, plan = plan, refIndex = 1, roi = ROI4D) # 4D method
    output_folder = os.path.join(output_path, 'Robustness4D_Test')
    scenarios.save(output_folder)


scenarios.analyzeErrorSpace(RefCT, "D95", RefTV, plan.planDesign.objectives.targetPrescription)
scenarios.printInfo()
scenarios.recomputeDVH([RefTV])

#%%
# Show results
#----------------------
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for i, dvh_band in enumerate(scenarios.dvhBands):
    color = f'C{i % 10}'
    phigh = ax.plot(dvh_band._dose, dvh_band._volumeHigh, alpha=0)
    plow = ax.plot(dvh_band._dose, dvh_band._volumeLow, alpha=0)
    pNominal = ax.plot(dvh_band._nominalDVH._dose, dvh_band._nominalDVH._volume, label=dvh_band._roiName, color = color)
    pfill = ax.fill_between(dvh_band._dose, dvh_band._volumeHigh, dvh_band._volumeLow, alpha=0.2, color=color)
ax.set_xlabel("Dose (Gy)")
ax.set_ylabel("Volume (%)")
plt.grid(True)
plt.legend()
plt.savefig(f'{output_path}/Dose4DEvaluation.png', format = 'png')
plt.show()