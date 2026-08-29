import numpy as np
import scipy.optimize as opt

def TOF(optVars: float, txCoordinates: np.ndarray, rxCoordinates: np.ndarray, blockLength: float, velocityBlock: float, velocityCouplingMedium: float, aTarget: float, bTarget: float) -> float:
    """
    Calculate the time of flight (TOF) for a signal traveling from a transmitter to a receiver, considering refraction and reflection at a target interface.

    Parameters:
    -----------
    optVars (np.float64): Optimization variables, including the x-coordinates of the refraction and reflection points.
    txCoordinates (np.ndarray[np.float64]): Coordinates of the transmitter (x, y).
    rxCoordinates (np.ndarray[np.float64]): Coordinates of the receiver (x, y).
    blockLength (np.float64): Length of the block.
    velocityBlock (np.float64): Velocity in the block.
    velocityCouplingMedium (np.float64): Velocity in the coupling medium.
    aTarget (np.float64): Slope of the target line.
    bTarget (np.float64): Intercept of the target line.

    Returns:
    --------
    np.float64: Total time of flight from the transmitter to the receiver, considering the path through the refraction and reflection points.
    """

    xRefraction1, xReflection, xRefraction2 = optVars

    pRefraction1 = np.array([xRefraction1, blockLength])
    pReflection = np.array([xReflection, aTarget*xReflection + bTarget])
    pRefraction2 = np.array([xRefraction2, blockLength])

    tofRefraction1 = np.linalg.norm(txCoordinates - pRefraction1) / velocityBlock
    tofReflection = (np.linalg.norm(pRefraction1 - pReflection) + np.linalg.norm(pReflection - pRefraction2)) / velocityCouplingMedium
    tofRefraction2 = np.linalg.norm(pRefraction2 - rxCoordinates) / velocityBlock

    return tofRefraction1 + tofReflection + tofRefraction2

def GenerateTOFMatrix(transducerElementsCoordinates: np.ndarray, blockLength: float, velocityBlock: float, velocityCouplingMedium: float, aTarget: float, bTarget: float, generateFullMatrix=True, tol=1e-7) -> np.ndarray:
    """
    Generate a matrix of time of flight (TOF) values for a set of transducer elements, considering refraction and reflection at a desired target interface (defined by: y = aTarget * x + bTarget).

    Parameters:
    -----------
    transducerElementsCoordinates (np.ndarray): Coordinates of the transducer elements (x, y).
    blockLength (float): Length of the block.
    velocityBlock (float): Velocity in the block.
    velocityCouplingMedium (float): Velocity in the coupling medium.
    aTarget (float): Slope of the target line.
    bTarget (float): Intercept of the target line.
    generateFullMatrix (bool): If True, generate a full TOF matrix; if False, generate only the diagonal elements.
    tol (float): Tolerance for the optimization process.

    Returns:
    --------
    np.ndarray: A matrix of TOF values, where each element (i, j) represents the time of flight from transducer element i to transducer element j, considering the path through the refraction and reflection points. If generateFullMatrix is False, only the diagonal elements are computed.
    """

    nElements = transducerElementsCoordinates.shape[0]

    if generateFullMatrix:
        tofMatrix = np.zeros((nElements, nElements))
        for i in range(nElements):
            txCoordinate = transducerElementsCoordinates[i]
            for j in range(nElements):
                rxCoordinate = transducerElementsCoordinates[j]
                initialGuess = [rxCoordinate[0], (rxCoordinate[0] + txCoordinate[0])/2, txCoordinate[0]]

                results = opt.minimize(TOF, initialGuess,
                                        args=(txCoordinate, rxCoordinate, blockLength, velocityBlock, velocityCouplingMedium, aTarget, bTarget), tol=tol)
                tofMatrix[i,j] = results.fun
    else:
        tofMatrix = np.zeros(nElements)
        for i in range(nElements):
            txCoordinate = transducerElementsCoordinates[i]
            initialGuess = [txCoordinate[0], txCoordinate[0], txCoordinate[0]]

            results = opt.minimize(TOF, initialGuess,
                                        args=(txCoordinate, txCoordinate, blockLength, velocityBlock, velocityCouplingMedium, aTarget, bTarget), tol=tol)
            tofMatrix[i] = results.fun

    return tofMatrix

def TOFCost(optVars: np.ndarray, transducerElementsCoordinates: np.ndarray, blockLength: float, velocityBlock: float, velocityCouplingMedium: float, measuredTof: np.ndarray, tol: float = 1e-7) -> float:
    """
    Calculate the cost function for the time of flight (TOF) optimization problem, which measures the difference between the simulated TOF matrix and the measured TOF matrix.

    Parameters:
    -----------
    optVars (np.ndarray): Optimization variables, including the slope and intercept of the target line.
    transducerElementsCoordinates (np.ndarray): Coordinates of the transducer elements (x, y).
    blockLength (float): Length of the block.
    velocityBlock (float): Velocity in the block.
    velocityCouplingMedium (float): Velocity in the coupling medium.
    measuredTof (np.ndarray): Measured TOF matrix.
    tol (float): Tolerance for the optimization process.

    Returns:
    --------
    float: The cost value, which is the root mean square error between the simulated TOF matrix and the measured TOF matrix.
    """

    a, b = optVars
    nElements = transducerElementsCoordinates.shape[0]

    if np.shape(measuredTof) == (nElements, nElements):
        simulatedTof = GenerateTOFMatrix(transducerElementsCoordinates, blockLength, velocityBlock, velocityCouplingMedium, a, b, generateFullMatrix=True, tol=tol)
    else:
        simulatedTof = GenerateTOFMatrix(transducerElementsCoordinates, blockLength, velocityBlock, velocityCouplingMedium, a, b, generateFullMatrix=False, tol=tol)

    return np.sqrt(np.sum((simulatedTof - measuredTof)**2))
