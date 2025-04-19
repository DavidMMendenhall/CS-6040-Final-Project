import numpy as np
import pyvista as pv
import json
import OpenVisus as ov
import os
from utility import convert_to_sphere
import pickle

# Set cache
os.environ['VISUS_CACHE'] = "./visus_can_be_deleted"

# Set color range for the colorbar between -0.3 and 0.3
color_range_min = -0.3
color_range_max = 0.3



# Load CO2 Mesh data
mesh_id = "./mesh_data/CO2_0-10269-50_-12"
point_data = np.fromfile(f"{mesh_id}_mesh", dtype=np.float32)
face_data = np.fromfile(f"{mesh_id}_faces", dtype=np.int32)
meta_data = None
with open(f"{mesh_id}_meta.json") as file:
    meta_data = json.load(file)

point_data = point_data.reshape((point_data.shape[0] // 3, 3))

face_offsets = []
point_offsets = []
face_offset = 0
point_offset = 0
for m in range(len(meta_data['face_offsets'])):
    face_offsets.append(meta_data['face_offsets'][m])
    point_offsets.append(meta_data['mesh_offsets'][m])
face_offsets.append(point_data.shape[0])
point_offsets.append(face_data.shape[0])

def getMeshes(timeStep, radius=1, relief=0.25):
    offset = timeStep * 6
    result = []
    for face in range(6):

        result.append(
            (
                convert_to_sphere(point_data[point_offsets[offset + face]:point_offsets[offset + face + 1]], face, radius, relief),
                face_data[face_offsets[offset+face]:face_offsets[offset+face+1]]
             )
        )
    return result

myMesh = getMeshes(0)

# vorticity data
vorticity_dim = 180
vorticity_data = np.fromfile("./vort_data/vorticities_z50_0-10269-50_180", dtype=np.float32)
vorticity_data = vorticity_data.reshape((vorticity_data.shape[0]//(vorticity_dim * vorticity_dim), vorticity_dim, vorticity_dim))

pl = pv.Plotter()
pl.set_background("#212121")

render_params = {
    "time": 0,
    "relief": 0.1,
    "CO2_opacity": 0.5,
    "show_path_lines": True,
    "pathline_length": 3,

}
def render():
    myMesh = getMeshes(int(render_params['time'] / 50), 1, render_params['relief'])
    faces = [pv.PolyData(myMesh[face][0], myMesh[face][1]) for face in range(6)]
    n = vorticity_dim
    quads = np.array([[(4, (x + y * n), ((x+1) + y * n), ((x+1) + (y+1) * n), (x + (y+1) * n)) for x in range(n-1)] for y in range(n-1) ]).flatten()

    for face in range(6):
        pl.add_mesh(faces[face], color="#00FFFF", show_edges=False, name=f"co2_face_{face}", opacity=render_params['CO2_opacity'])

        
        ui = np.linspace(0.0, 1.0, n)
        vi = np.linspace(0.0, 1.0, n)
        u, v = np.meshgrid(ui, vi)

        x = np.zeros_like(u)
        y = u
        z = v

        points = np.zeros((x.shape[0] * x.shape[1] , 3), dtype=np.float32)
        points[:, 0] = x.flatten()
        points[:, 1] = y.flatten()
        points[:, 2] = z.flatten()

        points = convert_to_sphere(points, face, 1, 0)
        pv_mesh = pv.PolyData(points, quads)
        pv_mesh.point_data['vorticity'] = vorticity_data[(face + int(render_params['time'] / 50) * 6), :, :].flatten()

        pl.add_mesh(pv_mesh, clim=[-0.3, 0.3], cmap='RdBu', name=f"vorticity_face_{face}")

    new_poly = make_polydata(int(render_params['time'] / 5) + 1)
    new_poly = offset_pathlines(new_poly)  # Apply offset to updated pathlines
    actor = pl.add_mesh(new_poly, color="#00FF00", line_width=3, name="path_lines")
    actor.SetVisibility(render_params["show_path_lines"])



def updateRenderParam(value, parameter):
    render_params[parameter] = value
    render()



### --------- Pathlines Render --------
# Pathlines Variables
FILENAME      = "pathline_cache/pathlines.pkl"
SHOW_EVERY_N  = 1     # only draw every Nth trajectory for speed

with open(FILENAME, "rb") as f:
    all_pathlines = pickle.load(f)
print(f"Loaded {len(all_pathlines)} pathlines") # Data looks like [x, y, z, t]

# Sample down to keep things responsive. Not actually required right now but useful for speed.
sampled = [
    all_pathlines[i]
    for i in range(0, len(all_pathlines), SHOW_EVERY_N)
    if len(all_pathlines[i]) >= 2
]

# Helper: build PolyData showing only the last TRAIL_LENGTH points up to t
def make_polydata(t):
    points = []
    lines  = []
    offset = 0

    for traj in sampled:
        pts = np.asarray(traj, dtype=np.float32)
        L   = pts.shape[0]
        t_idx = max(0, min(int(t), L - 1))
        start = max(0, t_idx - render_params["pathline_length"] + 1)
        seg = pts[start : t_idx + 1]
        if seg.shape[0] < 2:
            continue
        points.extend(seg)
        # VTK line cell: [n_pts, id0, id1, …]
        lines.append([seg.shape[0]] + list(range(offset, offset + seg.shape[0])))
        offset += seg.shape[0]

    if not points:
        return pv.PolyData()
    pts_arr  = np.asarray(points)
    lines_arr = np.hstack(lines)
    return pv.PolyData(pts_arr, lines=lines_arr)

# Add a small offset to pathlines' coordinates to ensure they are not obscured
# I have to do this because the sphere and lines are drawn at the same scale.
def offset_pathlines(pathline_data, offset_value=1.1):
    pathline_data.points *= offset_value  # Apply small offset to X-axis (or Y/Z if necessary)
    return pathline_data


# its only 51 timesteps because the dataset im using is quite small. 
# If we want to get a bigger set we need to generate new pathlines using pathlineCreation.py


render()

# pl.add_slider_widget(lambda value: updateRenderParam(int(value), "time"), pointa=(0.1, 0.9), pointb=(0.9, 0.9), rng=(0, 10269, 1), value=render_params ["time"], title="Time", color="white", fmt="%0.0f", style="modern")
pl.add_slider_widget(lambda value: updateRenderParam(int(value), "time"), pointa=(0, 0.9), pointb=(0.5, 0.9), rng=(0, 1000, 1), value=render_params ["time"], title="Time", color="white", fmt="%0.0f", style="modern")
pl.add_slider_widget(lambda value: updateRenderParam(value, "relief"), pointa=(0, 0.75), pointb=(0.2, 0.75), rng=(0, 1), value=render_params["relief"], title="CO2 Relief", color="#00FFFF", fmt="%0.2f", style="modern")
pl.add_slider_widget(lambda value: updateRenderParam(value, "CO2_opacity"), pointa=(0.0, 0.6), pointb=(0.2, 0.6), rng=(0, 1), value=render_params["CO2_opacity"], title="CO2 Opacity", color="#00FFFF", fmt="%0.2f", style="modern")
pl.add_checkbox_button_widget(lambda value: updateRenderParam(value, "show_path_lines"), value=render_params['show_path_lines'], position=(0.1, 0.1), color_on="#00FF00")
# pl.show_grid()
pl.show()