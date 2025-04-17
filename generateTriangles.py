import numpy as np

class CubeBaseCase():
    def __init__(self, base_cube_id: np.uint8, triangles: list[tuple[int, int, int]]):
        self.base_cube_id = base_cube_id
        self.triangles = triangles
        self.id = -1
    # Maps cube_id to base case
    TABLE = {}

CUBE_BASE_CASES = [
        CubeBaseCase(0b00000000, [(-1,-1,-1),(-1,-1,-1),(-1,-1,-1),(-1,-1,-1)]), #0
        CubeBaseCase(0b00000001, [(0,3,8), (-1,-1,-1),(-1,-1,-1),(-1,-1,-1)]), #1
        CubeBaseCase(0b00000011, [(1,3,9), (3,8,9), (-1,-1,-1), (-1,-1,-1)]), #2
        CubeBaseCase(0b00100001, [(0,3,8), (9,4,5), (-1,-1,-1), (-1,-1,-1)]), #3
        CubeBaseCase(0b01000001, [(0,3,8), (5,6,10), (-1,-1,-1), (-1,-1,-1)]), #4
        CubeBaseCase(0b00001110, [(0,3,9), (3,11,9), (11,10,9),(-1,-1,-1)]), #5
        CubeBaseCase(0b01000011, [(1,3,9), (3,8,9), (5,6,10), (-1,-1,-1)]), #6
        CubeBaseCase(0b01010010, [(0,9,1), (8,7,4), (5,9,10),(-1,-1,-1)]), #7
        CubeBaseCase(0b00001111, [(8,10,11), (8,9,10),(-1,-1,-1),(-1,-1,-1)]), #8
        CubeBaseCase(0b10001101, [(7,8,6), (6,8,0), (6,0,10), (10,0,1)]), #9
        CubeBaseCase(0b01010101, [(7,4,3), (4,0,3), (6,2,5), (5,2,1)]), #10
        CubeBaseCase(0b01001101, [(11,8,0), (11,0,5), (6,11,5), (5,0,1)]), #11
        CubeBaseCase(0b00011110, [(9,3,0), (11,3,9), (10,11,9), (7,4,8)]), #12
        CubeBaseCase(0b10100101, [(0,3,8), (1,10,2), (11,6,7), (9,4,5)]), #13
        CubeBaseCase(0b10001110, [(0,3,11), (0,7,10), (0,10,1), (7,6,10)]), #14
    ]

class CubeCase():
    def __init__(self, cube_id: np.uint8, base_case: CubeBaseCase, invert_base_case: bool, orientation: int):
       self.cube_id = cube_id
       self.base_case = base_case
       self.invert_base_case = invert_base_case
       self.orientation = orientation
    
    def getTriangles(self):
        triangles = [self.__convertTriangle(triangle) for triangle in self.base_case.triangles]
        return triangles
        

    def __str__(self):
        return f"id: {self.cube_id:08b}, base case: {self.base_case.base_cube_id:08b} #{self.base_case.id}, invert: {self.invert_base_case}, orientation: {self.orientation}"

    def __convertTriangle(self, triangle):
        if self.invert_base_case:
            triangle = CubeCase.__swapTriangleWinding(triangle)
        return CubeCase.__rotateTriangle(triangle, self.orientation)
        
    @staticmethod
    def __swapTriangleWinding(triangle: tuple[int, int, int]):
        return (triangle[0], triangle[2], triangle[1])
    
    __inverseRotateEdgeZTable = {
        -1: -1,
        0:  9,
        1:  5,
        2:  10,
        3:  1,
        4:  8,
        5:  7,
        6:  11,
        7:  3,
        8:  0,
        9:  4,
        10: 6,
        11: 2,
    }
    __inverseRotateEdgeYTable = {
        -1: -1,
        0:  1,
        1:  2,
        2:  3,
        3:  0,
        4:  5,
        5:  6,
        6:  7,
        7:  4,
        8:  9,
        9:  10,
        10: 11,
        11: 8,
    }
    __inverseRotateEdgeXTable = {
        -1: -1,
        2:  0,
        10: 1,
        6:  2,
        11: 3,
        0:  4,
        9:  5,
        4:  6,
        8:  7,
        3:  8,
        1:  9,
        5:  10,
        7:  11,
    }
    
    @staticmethod
    def __rotateTriangle(triangle: tuple[int, int, int], orientation: int) -> int:
        assert orientation < 4 * 6, "Invalid orientation value"
        y_count = orientation % 4
        axis_num = orientation // 4
        x_count = 1 if  axis_num == 1 else (3 if  axis_num == 2 else 0)
        z_count = 1 if  axis_num == 3 else (3 if  axis_num == 4 else (2 if  axis_num == 5 else 0))
        while y_count > 0:
            y_count -= 1
            triangle = (CubeCase.__inverseRotateEdgeYTable[triangle[0]], CubeCase.__inverseRotateEdgeYTable[triangle[1]], CubeCase.__inverseRotateEdgeYTable[triangle[2]])
        while x_count > 0:
            x_count -= 1
            triangle = (CubeCase.__inverseRotateEdgeXTable[triangle[0]], CubeCase.__inverseRotateEdgeXTable[triangle[1]], CubeCase.__inverseRotateEdgeXTable[triangle[2]])
        while z_count > 0:
            z_count -= 1
            triangle = (CubeCase.__inverseRotateEdgeZTable[triangle[0]], CubeCase.__inverseRotateEdgeZTable[triangle[1]], CubeCase.__inverseRotateEdgeZTable[triangle[2]])
        return triangle

def invertCube(cube_id: np.uint8) -> np.uint8: 
    return np.uint8(cube_id ^ 0b11111111)

def rotateCubeZ(cube_id: np.uint8) -> np.uint8:
    b0 = ((cube_id >> 1) & 0b1) << 0
    b1 = ((cube_id >> 5) & 0b1) << 1
    b2 = ((cube_id >> 6) & 0b1) << 2
    b3 = ((cube_id >> 2) & 0b1) << 3
    b4 = ((cube_id >> 0) & 0b1) << 4
    b5 = ((cube_id >> 4) & 0b1) << 5
    b6 = ((cube_id >> 7) & 0b1) << 6
    b7 = ((cube_id >> 3) & 0b1) << 7
    return np.uint8(b7 | b6 | b5 | b4 | b3 | b2 | b1 | b0)

def rotateCubeY(cube_id: np.uint8)-> np.uint8:
    b0 = ((cube_id >> 1) & 0b1) << 0
    b1 = ((cube_id >> 2) & 0b1) << 1
    b2 = ((cube_id >> 3) & 0b1) << 2
    b3 = ((cube_id >> 0) & 0b1) << 3
    b4 = ((cube_id >> 5) & 0b1) << 4
    b5 = ((cube_id >> 6) & 0b1) << 5
    b6 = ((cube_id >> 7) & 0b1) << 6
    b7 = ((cube_id >> 4) & 0b1) << 7
    return np.uint8(b7 | b6 | b5 | b4 | b3 | b2 | b1 | b0)

def rotateCubeX(cube_id: np.uint8) -> np.uint8:
    b0 = ((cube_id >> 4) & 0b1) << 0
    b1 = ((cube_id >> 5) & 0b1) << 1
    b2 = ((cube_id >> 1) & 0b1) << 2
    b3 = ((cube_id >> 0) & 0b1) << 3
    b4 = ((cube_id >> 7) & 0b1) << 4
    b5 = ((cube_id >> 6) & 0b1) << 5
    b6 = ((cube_id >> 2) & 0b1) << 6
    b7 = ((cube_id >> 3) & 0b1) << 7
    return np.uint8(b7 | b6 | b5 | b4 | b3 | b2 | b1 | b0)

def rotateCube(cube_id: np.uint8, orientation: int) -> np.uint8:
    assert orientation < 4 * 6, "Invalid orientation value"
    y_count = orientation % 4
    axis_num = orientation // 4
    x_count = 1 if  axis_num == 1 else (3 if  axis_num == 2 else 0)
    z_count = 1 if  axis_num == 3 else (3 if  axis_num == 4 else (2 if  axis_num == 5 else 0))
    while x_count > 0:
        x_count -= 1
        cube_id = rotateCubeX(cube_id)
    while z_count > 0:
        z_count -= 1
        cube_id = rotateCubeZ(cube_id)
    while y_count > 0:
        y_count -= 1
        cube_id = rotateCubeY(cube_id)
    return cube_id

def cubeRequiresInversion(cube_id: np.uint8) -> bool:
    return cube_id.bit_count() > 4

# Populate CubeBaseCase.TABLE
for i in range(len(CUBE_BASE_CASES)):
    cubeCase = CUBE_BASE_CASES[i]
    CubeBaseCase.TABLE[cubeCase.base_cube_id] = cubeCase
    cubeCase.id = i


cube_case_table = [None] * 256

for cube_id_ in range(256):
    cube_id = np.uint8(cube_id_)
    inverted = cubeRequiresInversion(cube_id)
    base_id = None
    base_candidate_id = cube_id if not inverted else invertCube(cube_id)
    for orientation in range(24):
        oriented_candidate_id = rotateCube(base_candidate_id, orientation)
        if oriented_candidate_id in CubeBaseCase.TABLE:
            base_id = oriented_candidate_id
            cube_case_table[cube_id] = CubeCase(cube_id,CubeBaseCase.TABLE[oriented_candidate_id], inverted, orientation).getTriangles()
            break
    if base_id is None:
        print(f"Failed to match cube with id 0b{cube_id:08b}")

cube_case_table = np.array(cube_case_table, dtype=np.int32)

if __name__ == "__main__":
    cube_case_table.tofile("./cube_case_table")
    