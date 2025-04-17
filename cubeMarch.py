import numpy as np
from generateTriangles import cube_case_table 

def get_point_index(edge: int, cell: tuple[int, int, int], dim: tuple[int, int, int]) -> int:
    """
    Gets the index of the point in a mesh given the edge number, cell location, dimensions of the grid,
    and offsets of the x, y, and z aligned edges.
    edge: [0,11] the edge of the cell tof ind
    cell: (int, int, int) the cell to consider
    dim: (int, int, int) the width, height, and depth of the grid
    returns: int, index of the point searched, 0 if bad data is provided for safety
    """
    x, y, z = cell
    w, h, d = dim
    x_member_size = w * (h + 1) * (d + 1)
    y_member_size = (w + 1) * h * (d + 1)
    z_member_size = (w + 1) * (h + 1) * d

    x_member_offset = 0
    y_member_offset = x_member_offset + x_member_size
    z_member_offset = y_member_offset + y_member_size
    match(edge):
        # x aligned members
        case(0):
            return x + (y + 0) * w + (z + 0) * (h + 1) * w + x_member_offset
        case(2):
            return x + (y + 0) * w + (z + 1) * (h + 1) * w + x_member_offset
        case(4):
            return x + (y + 1) * w + (z + 0) * (h + 1) * w + x_member_offset
        case(6):
            return x + (y + 1) * w + (z + 1) * (h + 1) * w + x_member_offset
        # y aligned members
        case(8):
            return (x + 0) + y * (w + 1) + (z + 0) * h * (w + 1) + y_member_offset
        case(9):
            return (x + 1) + y * (w + 1) + (z + 0) * h * (w + 1) + y_member_offset
        case(10):
            return (x + 1) + y * (w + 1) + (z + 1) * h * (w + 1) + y_member_offset
        case(11):
            return (x + 0) + y * (w + 1) + (z + 1) * h * (w + 1) + y_member_offset
        # z aligned members
        case(1):
            return (x + 1) + (y + 0) * (w + 1) + z * (h + 1) * (w + 1) + z_member_offset
        case(3):
            return (x + 0) + (y + 0) * (w + 1) + z * (h + 1) * (w + 1) + z_member_offset
        case(5):
            return (x + 1) + (y + 1) * (w + 1) + z * (h + 1) * (w + 1) + z_member_offset
        case(7):
            return (x + 0) + (y + 1) * (w + 1) + z * (h + 1) * (w + 1) + z_member_offset
    return 0

def cube_march_build_faces(values: list[float], iso_value: float, pos: tuple[int, int, int], dim: tuple[int, int, int]) -> list[int]:
    cube_id = 0
    for v in values[::-1]:
        cube_id <<= 1
        cube_id = cube_id | (iso_value < v)
    
    cube_case_triangles = cube_case_table[cube_id]
    faces = []
    used_points = set()
    for triangle in cube_case_triangles:
        if triangle[0] == -1:
            break
        used_points.add(get_point_index(triangle[0], pos, dim),)
        used_points.add(get_point_index(triangle[1], pos, dim),)
        used_points.add(get_point_index(triangle[2], pos, dim),)
        faces += [
            3,
            get_point_index(triangle[0], pos, dim),
            get_point_index(triangle[1], pos, dim),
            get_point_index(triangle[2], pos, dim)

        ]
    return faces, used_points

def cube_march(data_grid: np.ndarray, iso_value: float) -> tuple[np.ndarray, np.ndarray]:
    w, h, d = data_grid.shape
    w -= 1
    h -= 1
    d -= 1
    x_member_size = w * (h + 1) * (d + 1)
    y_member_size = (w + 1) * h * (d + 1)
    z_member_size = (w + 1) * (h + 1) * d

    dims = (w, h, d)

    mesh_points = np.zeros((x_member_size + y_member_size + z_member_size, 3), dtype=np.float32)
    faces = []
    usedPoints = {0}
    for x in range(w + 1):
        for y in range(h + 1):
            for z in range(d + 1):
                # compute coordinate of each point on the
                # lesser ends of the cell
                v1 = data_grid[x, y, z]
                p1 = np.array((x, y, z))
                if x < w:
                    point_index = get_point_index(0, (x, y, z), dims)
                    v2 = data_grid[x+1, y, z]
                    p2 = np.array((x+1, y, z))
                    t = (iso_value - v1) / (v2 - v1) if v1 != v2 else 0.5
                    t = max(0, min(t, 1))
                    mesh_points[point_index] = (p2 - p1) * t + p1

                if y < h:
                    point_index = get_point_index(8, (x, y, z), dims)
                    v2 = data_grid[x, y+1, z]
                    p2 = np.array((x, y+1, z))
                    t = (iso_value - v1) / (v2 - v1) if v1 != v2 else 0.5
                    t = max(0, min(t, 1))
                    mesh_points[point_index] = (p2 - p1) * t + p1

                if z < d:
                    point_index = get_point_index(3, (x, y, z), dims)
                    v2 = data_grid[x, y, z+1]
                    p2 = np.array((x, y, z+1))
                    t = (iso_value - v1) / (v2 - v1) if v1 != v2 else 0.5
                    t = max(0, min(t, 1))
                    mesh_points[point_index] = (p2 - p1) * t + p1

                if x >= w or y >= h or z >= d: # not a full cell, contains no cube
                    continue
                values = [
                    data_grid[x, y, z],
                    data_grid[x+1, y, z],
                    data_grid[x+1, y, z+1],
                    data_grid[x, y, z+1],

                    data_grid[x, y+1, z],
                    data_grid[x+1, y+1, z],
                    data_grid[x+1, y+1, z+1],
                    data_grid[x, y+1, z+1],
                ]
                new_faces, new_points = cube_march_build_faces(values, iso_value, (x, y, z), dims)
                faces += new_faces
                usedPoints = new_points | usedPoints
                
    mesh_points[:, 0] /= w
    mesh_points[:, 1] /= h
    mesh_points[:, 2] /= d
    if len(faces) == 0:
        faces = [3, 0, 0, 0]
    faces = np.array(faces, dtype=np.int32)
    
    # mesh optimization
    needed_points = [False] * (x_member_size + y_member_size + z_member_size)

    offset = 0
    for point in range((x_member_size + y_member_size + z_member_size)):
        if  point in usedPoints:
            needed_points[point] = True
            faces[faces == point] -= offset
        else:
            offset += 1

    return (mesh_points[needed_points], faces)
    

def func_to_grid(func, start, end, grid_shape):
    x_axis = np.linspace(start[0], end[0], grid_shape[0] + 1)
    y_axis = np.linspace(start[1], end[1], grid_shape[1] + 1)
    z_axis = np.linspace(start[2], end[2], grid_shape[2] + 1)
    sample_function_v = np.vectorize(func)
    XX, YY, ZZ = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
    grid_values = sample_function_v(XX, YY, ZZ)
    return grid_values



if __name__ == "__main__":
    import pyvista as pv
    def testFunc2(x, y, z):
        r = 2
        return (np.sqrt(x**2 + y**2) - r)**2 + z**2

    testData = func_to_grid(testFunc2, (-5, -5, -5), (5, 5, 5), (50, 50, 50))
    cubeMesh = cube_march(testData, 1)

    surface = pv.PolyData(cubeMesh[0], cubeMesh[1])
    pl = pv.Plotter()
    pl.set_background("#666699")
    pl.add_mesh(surface, color="#61B229", line_width=3, show_edges=False)
    pl.show()


