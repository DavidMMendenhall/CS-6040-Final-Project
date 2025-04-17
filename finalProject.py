import numpy as np
import pyvista as pv
import json
import OpenVisus as ov
import os
from utility import convert_to_sphere

# Set cache
os.environ['VISUS_CACHE'] = "./visus_can_be_deleted"

# vorticity Computations ===================
print("Computing Vorticity")

# Parameters
faces = range(6)
base_url = "https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/GEOS"
common_params = "?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco"
dx, dy = 4.0, 4.0
timestep = 100
z_level = 10

# Face orientation layout
face_layout = {
    0: {'center': (0, 0, 1), 'normal': 'z'},   # Front
    1: {'center': (1, 0, 0), 'normal': 'x'},   # Right
    4: {'center': (-1, 0, 0), 'normal': 'x'},  # Left
    3: {'center': (0, 0, -1), 'normal': 'z'},  # Back
    2: {'center': (0, 1, 0), 'normal': 'y'},   # Top
    5: {'center': (0, -1, 0), 'normal': 'y'},  # Bottom
}

# Load vorticities with orientation correction
vorticities = [None] * 6
for face in face_layout.keys():
    u_url = f"{base_url}/GEOS_U/u_face_{face}_depth_52_time_0_10269.idx{common_params}"
    v_url = f"{base_url}/GEOS_V/v_face_{face}_depth_52_time_0_10269.idx{common_params}"
    try:
        db_u = ov.LoadDataset(u_url)
        db_v = ov.LoadDataset(v_url)
        data_u = db_u.read(time=timestep, quality=-4, z=[z_level, z_level + 1])
        data_v = db_v.read(time=timestep, quality=-4, z=[z_level, z_level + 1])
        u, v = data_u[0, :, :], data_v[0, :, :]
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        vort = dvdx - dudy

        # Orientation corrections per face
        if face == 0:
            vort = np.rot90(vort, k=1)
        elif face == 1:
            vort = np.rot90(vort, k=1)
        elif face == 2:
            # vort = np.fliplr(vort)
            # vort = np.rot90(vort, k=)
            pass
        elif face == 3:
            vort = np.rot90(vort, k=2)
            # pass
            # vort = np.fliplr(vort)
        elif face == 4:
            vort = np.rot90(vort, k=2)
            # pass
        elif face == 5:
            vort = np.rot90(vort, k=1)


        vorticities[face] = vort
    except Exception as e:
        print(f"Face {face} error: {e}")

# Normalize vorticity for consistent color scale
all_vorticities = np.concatenate([vort.flatten() for vort in vorticities if vort is not None])
vorticity_min = np.min(all_vorticities)
vorticity_max = np.max(all_vorticities)

# Set color range for the colorbar between -0.3 and 0.3
color_range_min = -0.3
color_range_max = 0.3

# Cubed-sphere projection
def cubed_sphere_projection(face_idx, n):
    xi = np.linspace(-np.pi / 4, np.pi / 4, n)
    eta = np.linspace(-np.pi / 4, np.pi / 4, n)
    xi, eta = np.meshgrid(xi, eta)

    tan_xi = np.tan(xi)
    tan_eta = np.tan(eta)
    one = np.ones_like(tan_xi)

    if face_idx == 0:  # Front (+Z)
        x = tan_xi
        y = tan_eta
        z = one
    elif face_idx == 1:  # Right (+X)
        x = one
        y = tan_xi
        z = -tan_eta
    elif face_idx == 2:  # Top (+Y)
        x = tan_xi
        y = one
        z = -tan_eta
    elif face_idx == 3:  # Back (-Z)
        x = -tan_xi
        y = -tan_eta
        z = -one
    elif face_idx == 4:  # Left (-X)
        x = -one
        y = -tan_xi
        z = -tan_eta
    elif face_idx == 5:  # Bottom (-Y)
        x = -tan_xi
        y = -one
        z = -tan_eta
    else:
        raise ValueError("Invalid face index")

    r = np.sqrt(x**2 + y**2 + z**2)
    return x / r, y / r, z / r






# Load CO2 Mesh data
mesh_id = "CO2_0-10269-50_-12"
point_data = np.fromfile(f"./mesh_data/mesh_{mesh_id}", dtype=np.float32)
face_data = np.fromfile(f"./mesh_data/face_{mesh_id}", dtype=np.int32)
face_data[0::4] = 3 # fix a bug from the compressor
meta_data = None
with open(f"./mesh_data/meta_{mesh_id}.json") as file:
    meta_data = json.load(file)

point_data = point_data.reshape((point_data.shape[0] // 3, 3))

face_offsets = []
point_offsets = []
face_offset = 0
point_offset = 0
for m in range(len(meta_data['face_counts'])):
    face_offsets.append(face_offset)
    point_offsets.append(point_offset)
    face_offset += meta_data['face_counts'][m]
    point_offset += meta_data['point_counts'][m]
face_offsets.append(face_offset)
point_offsets.append(point_offset)

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

myMesh = getMeshes(20)


pl = pv.Plotter()
pl.set_background("#212121")

render_params = {
    "time": 0,
    "relief": 0.1,
    "CO2_opacity": 0,
    "vorticity_z": 0,
    "pathline_length": 5,
}
def render():
    myMesh = getMeshes(int(render_params['time'] / 50), 1, render_params['relief'])
    faces = [pv.PolyData(myMesh[face][0], myMesh[face][1]) for face in range(6)]
    for face in range(6):
        pl.add_mesh(faces[face], color="green", show_edges=False, name=f"co2_face_{face}", opacity=render_params['CO2_opacity'])

def updateRenderParam(value, parameter):
    render_params[parameter] = value
    render()


# pl.add_mesh(pv.Sphere(1), color="blue")
# ipywidgets.interact(
#     updateCO2,
#     time=(0, 200),
#     relief=(0.0, 1.0, 0.1)
# )

for face_idx, vort in enumerate(vorticities):
    if vort is None:
        continue

    n = vort.shape[0]
    # x, y, z = cubed_sphere_projection(face_idx, n)
    # points = np.zeros((x.shape[0], x.shape[1] , 3), dtype=np.float32)
    # grid = np.meshgrid((np.linspace(0.0, 0.0, 1), np.linspace(0.0, 1, n), np.linspace(0.0, 1, n)))
    # grid = convert_to_sphere(grid, face_idx, 1, 0)
    # x = grid[:, 0]
    # y = grid[:, 1]
    # z = grid[:, 2]

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

    points = convert_to_sphere(points, face_idx, 1, 0)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    daMesh = pv.StructuredGrid(x, y, z)
    # daMesh.points = points
    daMesh.point_data['vorticity'] = vort.flatten()
    pl.add_mesh(daMesh, clim=[-0.3, 0.3], cmap='RdBu')
    # fig.add_trace(go.Surface(
    #     x=x, y=y, z=z,
    #     surfacecolor=vort,
    #     colorscale='RdBu',
    #     cmin=color_range_min, cmax=color_range_max,  # Set color range
    #     showscale=True,
    #     colorbar=dict(
    #         title="Vorticity",
    #         tickvals=[color_range_min, 0, color_range_max],  # Show ticks at -0.3, 0, 0.3
    #         ticktext=[str(color_range_min), "0", str(color_range_max)],
    #         ticks="outside",
    #         ticklen=5
    #     ),
    #     name=f"Face {face_idx}"
    # ))


render()
pl.add_slider_widget(lambda value: updateRenderParam(int(value), "time"), pointa=(0.1, 0.9), pointb=(0.9, 0.9), rng=(0, 10269, 1), value=render_params ["time"], title="Time", color="white", fmt="%0.0f", style="modern")
pl.add_slider_widget(lambda value: updateRenderParam(value, "relief"), pointa=(0.1, 0.8), pointb=(0.4, 0.8), rng=(0, 1), value=render_params["relief"], title="Relief", color="white", fmt="%0.2f", style="modern")
pl.add_slider_widget(lambda value: updateRenderParam(value, "CO2_opacity"), pointa=(0.1, 0.7), pointb=(0.4, 0.7), rng=(0, 1), value=render_params["CO2_opacity"], title="CO2_opacity", color="white", fmt="%0.2f", style="modern")
pl.show_grid()
pl.show()