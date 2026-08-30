import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from framework import file_m2k
import scipy.signal as signal
from CostFunctions import *

# Load the M2M Panther data using the mini-auspex package
# Change the path to the location of your .m2k file
data = file_m2k.read('examples/5deg_63mm.m2k', freq_transd=5,
                     bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

# Create B-scan visualization to identify the region of interest (ROI)
roiStart = 7000
roiEnd = 8000
b_scan = np.zeros((data.ascan_data.shape[0], data.ascan_data.shape[2]),
                  dtype=float)
for i in range(b_scan.shape[1]):
    b_scan[roiStart:roiEnd, i] = data.ascan_data[roiStart:roiEnd, i, i, 0]

# Display B-scan (log scale)
bScan = plt.figure(figsize=(10,5))
plt.imshow(np.log10(np.abs(b_scan[:,:])+1e0), aspect='auto')
plt.ylim([roiEnd, roiStart])
plt.colorbar()
plt.xlabel('n-th array element')
plt.ylabel('Sample')
plt.title('B-scan')

# Find the times for each peak in the A-scan data within the ROI and plot a point in the B-scan for each peak
peakTimes = np.zeros((data.ascan_data.shape[1], data.ascan_data.shape[2]))
for i in range(peakTimes.shape[0]):
    for j in range(peakTimes.shape[1]):
        aScan = data.ascan_data[:, i, j, 0]
        aScan = aScan - np.mean(aScan)
        hilbert = signal.hilbert(aScan)
        idx = np.argmax(np.abs(hilbert[roiStart:roiEnd])) + roiStart
        peakTimes[i,j] = data.time_grid[idx][0] * 1e-6
        if i == j:
            plt.plot(i, idx, 'ro')

# Define the transducer elements coordinates and the target line parameters
numberOfElements = 64
elementsPitch = 0.6*1e-3 # m
transducerLength = (numberOfElements - 1) * elementsPitch # m
elements = np.zeros((numberOfElements, 2))
elements[:, 0] = -transducerLength/2 + np.arange(numberOfElements) * elementsPitch # x
elements[:, 1] = 0 # y

# Define the block length
blockLength = 24e-3

# Define the velocities in the block and coupling medium
velocityBlock = 5900 # m/s
velocityCouplingMedium = 1480 # m/s

# Obtain the target line parameters (slope and intercept) from the measurements
initialGuess = [0, -blockLength - 1e-3] # assumes 0 degree angle and a distance of 1 mm from the block
results = opt.minimize(TOFCost, initialGuess, args=(elements, -blockLength, velocityBlock, velocityCouplingMedium, np.diag(peakTimes)))
simulationA = results.x[0]
simulationB = results.x[1]

# Plot the results of the optimization and the measured TOF values
comparativePlot = plt.figure(figsize=(10, 5))
plt.title("Comparative Plot of Measured and Simulated TOF")

# Block
blockWidth = 2.5*transducerLength
blockX1 = -blockWidth/2
blockX2 = blockWidth/2
blockY1 = -blockLength
blockY2 = -blockLength

# Define the real target line parameters (slope and intercept)
targetA = np.tan(np.deg2rad(5)) # slope corresponding to 5 degrees
targetB = -63.0e-3
targetSize = 1.2*blockWidth # m
targetX1 = -targetSize/2
targetX2 = targetSize/2
y1Alvo = targetX1*targetA + targetB
y2Alvo = targetX2*targetA + targetB

# reusltados da otimização
simulationX1 = -targetSize/2
simulationX2 = targetSize/2
simulationY1 = simulationA*simulationX1 + simulationB
simulationY2 = simulationA*simulationX2 + simulationB

for j in range(0, numberOfElements, 7):
    plt.plot(elements[j, 0], elements[j, 1], 'rs')
block = plt.Rectangle((-blockWidth/2, 0), blockWidth, -blockLength, fill=True, edgecolor='k', facecolor='gray')
plt.gca().add_patch(block)
plt.plot([targetX1, targetX2], [y1Alvo, y2Alvo], 'k-', label='Target')
plt.plot([simulationX1, simulationX2], [simulationY1, simulationY2], 'r--', label='Simulation')
plt.legend()

plt.show()
