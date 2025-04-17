import numpy as np
def convert_to_sphere(points, face, radius, relief):
    translatedPoints = points.copy()
    translatedPoints[:, 0] = 0.5
    translatedPoints[:, 1] -= 0.5
    translatedPoints[:, 2] -= 0.5
    lengths = np.linalg.norm(translatedPoints, axis=1, keepdims=True)
    unitVectors = np.nan_to_num(translatedPoints / lengths)

    radii = points.copy()
    radii[:, 1] = radii[:, 0]
    radii[:, 2] = radii[:, 0]
    radii *= relief
    radii += radius

    transformMat = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=np.float32)
    match face:
        # x+
        case 0:  
            transformMat = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float32)
        # z-
        case 1:  
            transformMat = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ], dtype=np.float32)    
        # y+
        case 2:  
            transformMat = np.array([
                [0, 0, -1],
                [1, 0, 0],
                [0, -1, 0]
        ], dtype=np.float32)
        # x-
        case 3:  
            transformMat = np.array([
                [-1, 0, 0],
                [0, 0, -1],
                [0, -1, 0]
            ], dtype=np.float32)
        # z+
        case 4:  
            transformMat = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ], dtype=np.float32)  
        # y-
        case 5:  
            transformMat = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ], dtype=np.float32)

            

    # assume up is x
    finalPoints = unitVectors * radii
    finalPoints[:] = finalPoints.dot(transformMat.T)
    return finalPoints
