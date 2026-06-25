

# standard numerics and linear algebra libraries
import numpy as np  
from Adva_Quant_Inform.operators import rotation 

def blochVisualization(gate):
    # sample some points in the xz plane and see where they go
    ini = np.array([1,0])
    axis = [0,1,0]

    angles = np.linspace(0,2*pi,20, endpoint=False)
    sampleStates = []
    samplePoints = []
    for phi in angles:
        state = rotation(axis, phi) @ ini
        sampleStates.append( state )
        samplePoints.append( state.conj().T @ [X,Y,Z] @ state )

    samplePointsRot = []
    for state in sampleStates:
        stateRot = gate @ state
        samplePointsRot.append( stateRot.conj().T @ [X,Y,Z] @ stateRot )s
