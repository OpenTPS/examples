'''
PBS Delivery Timings 
====================
author: OpenTPS team

This example will present the basis of PBS delivery timings with openTPS core.

running time: ~ 5 minutes
'''
#%% 
# Setting up the environment in google collab
#--------------------------------------------
import sys
if "google.colab" in sys.modules:
    from IPython import get_ipython
    get_ipython().system('git clone https://gitlab.com/openmcsquare/opentps.git')
    get_ipython().system('pip install ./opentps')
    get_ipython().system('pip install scipy==1.10.1')
    import opentps

#%%
#imports
import numpy as np
np.random.seed(42)

#%%
#import the needed opentps.core packages

from opentps.core.data.plan._planProtonBeam import PlanProtonBeam
from opentps.core.data.plan._planProtonLayer import PlanProtonLayer
from opentps.core.data.plan._protonPlan import ProtonPlan
from opentps.core.processing.planDeliverySimulation.scanAlgoBeamDeliveryTimings import ScanAlgoBeamDeliveryTimings
from opentps.core.processing.planDeliverySimulation.simpleBeamDeliveryTimings import SimpleBeamDeliveryTimings
from opentps.core.io.dicomIO import readDicomPlan

#%%
# Create random plan
#-------------------
plan = ProtonPlan()
plan.appendBeam(PlanProtonBeam())
energies = np.array([130, 140, 150, 160, 170])
for m in energies:
    layer = PlanProtonLayer(m)
    x = 10*np.random.random(5) - 5
    y = 10*np.random.random(5) - 5
    mu = 5*np.random.random(5)

    layer.appendSpot(x, y, mu)
    plan.beams[0].appendLayer(layer)


bdt = SimpleBeamDeliveryTimings(plan)
plan_with_timings = bdt.getPBSTimings(sort_spots="true")

#%%
# Print plan
#-----------
print(plan_with_timings._beams[0]._layers[0].__dict__)