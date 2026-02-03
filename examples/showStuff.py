from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import binary_dilation
import numpy as np

from opentps.core.processing.doseCalculation.doseCalculationConfig import DoseCalculationConfig
from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import MCsquareDoseCalculator
from opentps.core.io import mcsquareIO
from opentps.core.io.scannerReader import readScanner
from opentps.core.data import DVH
from opentps.core.processing.planOptimization.tools import evaluateClinical
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D, resampleImage3D

def showModelWithAnimatedFields(model):

    for field in model.deformationList:
        field.resample(spacing=model.midp.spacing, gridSize=model.midp.gridSize, origin=model.midp.origin)

    y_slice = int(model.midp.gridSize[1] / 2)

    plt.figure()
    fig = plt.gcf()

    def updateAnim(imageIndex):
        fig.clear()
        compX = model.deformationList[imageIndex].velocity.imageArray[:, y_slice, :, 0]
        compZ = model.deformationList[imageIndex].velocity.imageArray[:, y_slice, :, 2]
        plt.imshow(model.midp.imageArray[:, y_slice, :][::5, ::5], cmap='gray')
        plt.quiver(compZ[::5, ::5], compX[::5, ::5], alpha=0.2, color='red', angles='xy', scale_units='xy', scale=5)

    anim = FuncAnimation(fig, updateAnim, frames=len(model.deformationList), interval=300)

    # anim.save('D:/anim.gif')
    plt.show()


def show2DMaskBorder(filledMaskSlice, color='red'):

    dilatedROI = binary_dilation(filledMaskSlice)
    border = np.logical_xor(dilatedROI, filledMaskSlice)

    return border


def simulateAndShowOptiDoseDVH(ct, plan, roi, refSolver, figName='', outPutPath='', show=True):
    # MCsquare simulation
    ctCalibration = readScanner(DoseCalculationConfig().scannerFolder)
    bdl = mcsquareIO.readBDL(DoseCalculationConfig().bdlFile)

    # Configure MCsquare
    mc2 = MCsquareDoseCalculator()
    mc2.beamModel = bdl
    mc2.ctCalibration = ctCalibration
    mc2.nbPrimaries = 1e7
    doseImage = mc2.computeDose(ct, plan)

    # Compute DVH on resampled contour
    target_DVH = DVH(roi, doseImage)
    # lung_DVH = DVH()
    print('D5 - D95 =  {} Gy'.format(target_DVH.D5 - target_DVH.D95))
    clinROI = [roi.name, roi.name]
    clinMetric = ["Dmin", "Dmax"]
    clinLimit = [59., 61.]
    clinObj = {'ROI': clinROI, 'Metric': clinMetric, 'Limit': clinLimit}
    print('Clinical evaluation')
    evaluateClinical(doseImage, [roi], clinObj)

    # center of mass
    roi = resampleImage3DOnImage3D(roi, ct)
    COM_coord = roi.centerOfMass
    COM_index = roi.getVoxelIndexFromPosition(COM_coord)
    X_coord = COM_index[0]
    Y_coord = COM_index[1]
    Z_coord = COM_index[2]

    contourTargetMask = roi.getBinaryContourMask()

    img_ct1 = ct.imageArray[:, :, Z_coord].transpose(1, 0)
    img_mask1 = contourTargetMask.imageArray[:, :, Z_coord].transpose(1, 0)
    img_dose1 = resampleImage3DOnImage3D(doseImage, ct)
    img_dose1 = img_dose1.imageArray[:, :, Z_coord].transpose(1, 0)

    img_ct2 = ct.imageArray[X_coord, :, :].transpose(1, 0)
    img_mask2 = contourTargetMask.imageArray[X_coord, :, :].transpose(1, 0)
    img_dose2 = resampleImage3DOnImage3D(doseImage, ct)
    img_dose2 = img_dose2.imageArray[X_coord, :, :].transpose(1, 0)

    img_ct3 = ct.imageArray[:, Y_coord, :].transpose(1, 0)
    img_mask3 = contourTargetMask.imageArray[:, Y_coord, :].transpose(1, 0)
    img_dose3 = resampleImage3DOnImage3D(doseImage, ct)
    img_dose3 = img_dose3.imageArray[:, Y_coord, :].transpose(1, 0)

    # Display dose
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(figName)

    ax[0, 0].axes.get_xaxis().set_visible(False)
    ax[0, 0].axes.get_yaxis().set_visible(False)
    ax[0, 0].imshow(img_ct1, cmap='gray')
    ax[0, 0].imshow(img_mask1, alpha=.2, cmap='binary')  # PTV
    dose = ax[0, 0].imshow(img_dose1, cmap='jet', alpha=.2)
    plt.colorbar(dose, ax=ax[0, 0])

    ax[0, 1].axes.get_xaxis().set_visible(False)
    ax[0, 1].axes.get_yaxis().set_visible(False)
    ax[0, 1].imshow(img_ct2, cmap='gray')
    ax[0, 1].imshow(img_mask2, alpha=.2, cmap='binary')  # PTV
    dose = ax[0, 1].imshow(img_dose2, cmap='jet', alpha=.2)
    plt.colorbar(dose, ax=ax[0, 1])

    ax[0, 2].axes.get_xaxis().set_visible(False)
    ax[0, 2].axes.get_yaxis().set_visible(False)
    ax[0, 2].imshow(img_ct3, cmap='gray')
    ax[0, 2].imshow(img_mask3, alpha=.2, cmap='binary')  # PTV
    dose = ax[0, 2].imshow(img_dose3, cmap='jet', alpha=.2)
    plt.colorbar(dose, ax=ax[0, 2])

    ax[1, 0].plot(target_DVH.histogram[0], target_DVH.histogram[1], label=target_DVH.name)
    ax[1, 0].set_xlabel("Dose (Gy)")
    ax[1, 0].set_ylabel("Volume (%)")
    ax[1, 0].grid(True)
    ax[1, 0].legend()

    convData = refSolver.getConvergenceData()
    # print("convData['time']", convData['time'])
    # print("convData['nIter']", convData['nIter'])
    # print("convData['func_0']", len(convData['func_0']))
    # print('i had the issue here with len(convData[func_0]) = 215 --> odd number')
    # print(len(np.arange(0, convData['time'], convData['time'] / convData['nIter'])))
    timeAxis = np.arange(0, convData['time'], convData['time'] / convData['nIter'])
    if len(timeAxis) > len(convData['func_0']): timeAxis = timeAxis[:-1]
    ax[1, 1].plot(timeAxis, convData['func_0'], 'bo-', lw=2, label='Fidelity')
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 1].set_ylabel('Cost')
    ax[1, 1].set_yscale('symlog')
    ax4 = ax[1, 1].twiny()
    ax4.set_xlabel('Iterations')
    ax4.set_xlim(0, convData['nIter'])
    ax[1, 1].grid(True)
    plt.savefig(outPutPath + figName)
    if show:
        plt.show()

def simulateAndShowDoseDVH(ct, plan, roi, figName='', outPutPath='', show=True):
    # MCsquare simulation
    ctCalibration = readScanner(DoseCalculationConfig().scannerFolder)
    bdl = mcsquareIO.readBDL(DoseCalculationConfig().bdlFile)

    # Configure MCsquare
    mc2 = MCsquareDoseCalculator()
    mc2.beamModel = bdl
    mc2.nbPrimaries = 5e4
    mc2.ctCalibration = ctCalibration
    mc2.nbPrimaries = 1e7
    doseImage = mc2.computeDose(ct, plan)

    # Compute DVH on resampled contour
    target_DVH = DVH(roi, doseImage)
    # lung_DVH = DVH()
    print('D5 - D95 =  {} Gy'.format(target_DVH.D5 - target_DVH.D95))
    clinROI = [roi.name, roi.name]
    clinMetric = ["Dmin", "Dmax"]
    clinLimit = [59., 61.]
    clinObj = {'ROI': clinROI, 'Metric': clinMetric, 'Limit': clinLimit}
    print('Clinical evaluation')
    evaluateClinical(doseImage, [roi], clinObj)

    # center of mass
    roi = resampleImage3DOnImage3D(roi, ct)
    COM_coord = roi.centerOfMass
    COM_index = roi.getVoxelIndexFromPosition(COM_coord)
    X_coord = COM_index[0]
    Y_coord = COM_index[1]
    Z_coord = COM_index[2]

    contourTargetMask = roi.getBinaryContourMask()

    img_ct1 = ct.imageArray[:, :, Z_coord].transpose(1, 0)
    img_mask1 = contourTargetMask.imageArray[:, :, Z_coord].transpose(1, 0)
    img_dose1 = resampleImage3DOnImage3D(doseImage, ct)
    img_dose1 = img_dose1.imageArray[:, :, Z_coord].transpose(1, 0)

    img_ct2 = np.rot90(ct.imageArray[X_coord, :, :])
    img_mask2 = np.rot90(contourTargetMask.imageArray[X_coord, :, :])
    img_dose2 = resampleImage3DOnImage3D(doseImage, ct)
    img_dose2 = np.rot90(img_dose2.imageArray[X_coord, :, :])

    img_ct3 = np.rot90(ct.imageArray[:, Y_coord, :])
    img_mask3 = np.rot90(contourTargetMask.imageArray[:, Y_coord, :])
    img_dose3 = resampleImage3DOnImage3D(doseImage, ct)
    img_dose3 = np.rot90(img_dose3.imageArray[:, Y_coord, :])

    # Display dose
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(figName)
    # ax[0, 0].axes.get_xaxis().set_visible(False)
    ax[0, 0].axes.get_yaxis().set_visible(False)
    ax[0, 0].imshow(img_ct1, cmap='gray')
    ax[0, 0].imshow(img_mask1, alpha=.2, cmap='binary')  # PTV
    ax[0, 0].set_xlabel('Axial')
    dose = ax[0, 0].imshow(img_dose1, cmap='jet', alpha=.2)
    plt.colorbar(dose, ax=ax[0, 0])

    # ax[0, 1].axes.get_xaxis().set_visible(False)
    ax[0, 1].axes.get_yaxis().set_visible(False)
    ax[0, 1].imshow(img_ct2, cmap='gray')
    ax[0, 1].imshow(img_mask2, alpha=.2, cmap='binary')  # PTV
    ax[0, 1].set_xlabel('Sagittal')
    dose = ax[0, 1].imshow(img_dose2, cmap='jet', alpha=.2)
    plt.colorbar(dose, ax=ax[0, 1])

    # ax[1, 0].axes.get_xaxis().set_visible(False)
    ax[1, 0].axes.get_yaxis().set_visible(False)
    ax[1, 0].imshow(img_ct3, cmap='gray')
    ax[1, 0].imshow(img_mask3, alpha=.2, cmap='binary')  # PTV
    ax[1, 0].set_xlabel('Coronal')
    dose = ax[1, 0].imshow(img_dose3, cmap='jet', alpha=.2)
    plt.colorbar(dose, ax=ax[1, 0])

    ax[1, 1].plot(target_DVH.histogram[0], target_DVH.histogram[1], label=target_DVH.name)
    ax[1, 1].set_xlabel("Dose (Gy)")
    ax[1, 1].set_ylabel("Volume (%)")
    ax[1, 1].grid(True)
    ax[1, 1].legend()

    plt.savefig(outPutPath + figName)
    if show:
        plt.show()

    return doseImage

def showDoseAndDVH(ct, doseImage, roi, figName='', outPutPath='', show=True):

    # Compute DVH on resampled contour
    target_DVH = DVH(roi, doseImage)
    # lung_DVH = DVH()
    print('D5 - D95 =  {} Gy'.format(target_DVH.D5 - target_DVH.D95))
    clinROI = [roi.name, roi.name]
    clinMetric = ["Dmin", "Dmax"]
    clinLimit = [59., 61.]
    clinObj = {'ROI': clinROI, 'Metric': clinMetric, 'Limit': clinLimit}
    print('Clinical evaluation')
    evaluateClinical(doseImage, [roi], clinObj)

    # center of mass
    roi = resampleImage3DOnImage3D(roi, ct)
    COM_coord = roi.centerOfMass
    COM_index = roi.getVoxelIndexFromPosition(COM_coord)
    X_coord = COM_index[0]
    Y_coord = COM_index[1]
    Z_coord = COM_index[2]

    contourTargetMask = roi.getBinaryContourMask()

    img_ct1 = ct.imageArray[:, :, Z_coord].transpose(1, 0)
    img_mask1 = contourTargetMask.imageArray[:, :, Z_coord].transpose(1, 0)
    img_dose1 = resampleImage3DOnImage3D(doseImage, ct)
    img_dose1 = img_dose1.imageArray[:, :, Z_coord].transpose(1, 0)

    img_ct2 = np.rot90(ct.imageArray[X_coord, :, :])
    img_mask2 = np.rot90(contourTargetMask.imageArray[X_coord, :, :])
    img_dose2 = resampleImage3DOnImage3D(doseImage, ct)
    img_dose2 = np.rot90(img_dose2.imageArray[X_coord, :, :])

    img_ct3 = np.rot90(ct.imageArray[:, Y_coord, :])
    img_mask3 = np.rot90(contourTargetMask.imageArray[:, Y_coord, :])
    img_dose3 = resampleImage3DOnImage3D(doseImage, ct)
    img_dose3 = np.rot90(img_dose3.imageArray[:, Y_coord, :])

    # Display dose
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(figName)
    # ax[0, 0].axes.get_xaxis().set_visible(False)
    ax[0, 0].axes.get_yaxis().set_visible(False)
    ax[0, 0].imshow(img_ct1, cmap='gray')
    ax[0, 0].imshow(img_mask1, alpha=.2, cmap='binary')  # PTV
    ax[0, 0].set_xlabel('Axial')
    dose = ax[0, 0].imshow(img_dose1, cmap='jet', alpha=.2)
    plt.colorbar(dose, ax=ax[0, 0])

    # ax[0, 1].axes.get_xaxis().set_visible(False)
    ax[0, 1].axes.get_yaxis().set_visible(False)
    ax[0, 1].imshow(img_ct2, cmap='gray')
    ax[0, 1].imshow(img_mask2, alpha=.2, cmap='binary')  # PTV
    ax[0, 1].set_xlabel('Sagittal')
    dose = ax[0, 1].imshow(img_dose2, cmap='jet', alpha=.2)
    plt.colorbar(dose, ax=ax[0, 1])

    # ax[1, 0].axes.get_xaxis().set_visible(False)
    ax[1, 0].axes.get_yaxis().set_visible(False)
    ax[1, 0].imshow(img_ct3, cmap='gray')
    ax[1, 0].imshow(img_mask3, alpha=.2, cmap='binary')  # PTV
    ax[1, 0].set_xlabel('Coronal')
    dose = ax[1, 0].imshow(img_dose3, cmap='jet', alpha=.2)
    plt.colorbar(dose, ax=ax[1, 0])

    ax[1, 1].plot(target_DVH.histogram[0], target_DVH.histogram[1], label=target_DVH.name)
    ax[1, 1].set_xlabel("Dose (Gy)")
    ax[1, 1].set_ylabel("Volume (%)")
    ax[1, 1].grid(True)
    ax[1, 1].legend()

    plt.savefig(outPutPath + figName)
    if show:
        plt.show()

    return doseImage

def show3ViewsOnTarget(image, targetCOMInVoxel, imgName='', savingPath='', show=False):
    fig, axs = plt.subplots(1, 3)

    fig.suptitle('3 plane view of ' + imgName + ' centered on target')

    axs[0].imshow(np.rot90(image.imageArray[:, targetCOMInVoxel[1], :]), cmap='gray')
    axs[0].set_title('Coronal View')
    axs[1].imshow(np.rot90(image.imageArray[targetCOMInVoxel[0], :, :]), cmap='gray')
    axs[1].set_title('Sagittal View')
    axs[2].imshow(np.rot90(image.imageArray[:, :, targetCOMInVoxel[2]], 3), cmap='gray')
    axs[2].set_title('Axial View')

    if savingPath:
        plt.savefig(savingPath + imgName + 'Target3DView')
    if show:
        plt.show()
