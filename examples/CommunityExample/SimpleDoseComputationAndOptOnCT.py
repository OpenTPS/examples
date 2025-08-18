'''
Simple dose computation and optimization on a real CT image
====================================
author: Eliot Peeters

In this example we are going to see how to :
- Import real dicom images and RT struct
- Create a plan
- Compute beamlets
- Optimize a plan with beamlets
- Save a plan and beamlets
- Compute DVH histograms
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
import numpy as np
import os
from matplotlib import pyplot as plt

#%%
#import the needed opentps.core packages

from opentps.core.data.plan import ProtonPlanDesign
from opentps.core.data import DVH
from opentps.core.io import mcsquareIO
from opentps.core.io.scannerReader import readScanner
from opentps.core.processing.doseCalculation.doseCalculationConfig import DoseCalculationConfig
from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import MCsquareDoseCalculator
from opentps.core.io.dataLoader import readData
from opentps.core.data.plan import ObjectivesList
from opentps.core.data.plan import FidObjective
from opentps.core.io.serializedObjectIO import saveBeamlets, saveRTPlan, loadBeamlets, loadRTPlan

#%%
#In the next cell we configure the CT scan model used for the dose calculation and the bdl model. The ones used in this example are the default configuration of openTPS wich may lead to some imprecision.

ctCalibration = readScanner(DoseCalculationConfig().scannerFolder)
bdl = mcsquareIO.readBDL(DoseCalculationConfig().bdlFile)

#%%
#Data importation
#----------------
#The dataset used in this example comes from the `Proknow website <https://proknowsystems.com/planning/studies/5a0f6aa074403fbcc665424c1b13eaf2/instructions>`_, 2018 TROG Plan Study: SRS Brain. The readData functions automatically import the subfolders and detects the type of data (CT or RT_struct).

ctImagePath = "./data" #The folder is initially named 'data'
data = readData(ctImagePath)

import os

print("Contenu de ./data :")
for root, dirs, files in os.walk(ctImagePath):
    print(root, ":", files)


rt_struct = data[0]
ct = data[1]