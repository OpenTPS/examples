"""
OpenTPS CORE : Real case optimization
-------------------------------------
*Author* : Eliot Peeters

"""

##############################################
#I am writing stuff
##############################################

import numpy as np
import os
from matplotlib import pyplot as plt

#Import the needed opentps.core packages

from opentps.core.data.plan import PlanDesign
from opentps.core.data import DVH
from opentps.core.io import mcsquareIO
from opentps.core.io.scannerReader import readScanner
from opentps.core.processing.doseCalculation.doseCalculationConfig import DoseCalculationConfig
from opentps.core.processing.doseCalculation.mcsquareDoseCalculator import MCsquareDoseCalculator
from opentps.core.io.dataLoader import readData
from opentps.core.data.plan import ObjectivesList
from opentps.core.data.plan import FidObjective
from opentps.core.io.serializedObjectIO import saveBeamlets, saveRTPlan, loadBeamlets, loadRTPlan



# In the next cell we configure the CT scan model used for the dose calculation and the bdl model. The ones used in this example are the default configuration of openTPS which may lead to some imprecision.

ctCalibration = readScanner(DoseCalculationConfig().scannerFolder)
bdl = mcsquareIO.readBDL(DoseCalculationConfig().bdlFile)