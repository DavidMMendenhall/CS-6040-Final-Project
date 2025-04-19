import numpy as np
import OpenVisus as ov
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

# ------------------------------------------------------------------------------
# Configuration and Setup
# ------------------------------------------------------------------------------

os.environ['VISUS_CACHE'] = "./visus_can_be_deleted"
cache_dir = "pathline_cache"
os.makedirs(cache_dir, exist_ok=True)

variable_u, variable_v = 'u', 'v'
faces = range(6)
base_url = "https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/GEOS"
common_params = "?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco"
z_level = 50
timesteps = list(range(0, 10269, 50))
grid_shape = (48, 48)
particles_per_face = 1000

# ------------------------------------------------------------------------------
# Utility Functions for Spherical Mapping and Particle Seeding
# ------------------------------------------------------------------------------

def face_to_sphere(face, x, y):
    a = 2 * x / (x.shape[1] - 1) - 1
    b = 2 * y / (y.shape[0] - 1) - 1
    mapping = [
        (np.ones_like(a), -b, -a),
        (a, -np.ones_like(b), -b),
        (-np.ones_like(a), b, -a),
        (-a, np.ones_like(b), -b),
        (a, -b, np.ones_like(a)),
        (-a, -b, -np.ones_like(a)),
    ]
    x3, y3, z3 = mapping[face]
    norm = np.sqrt(x3**2 + y3**2 + z3**2)
    return x3 / norm, y3 / norm, z3 / norm

def compute_tangent_vectors(x, y, z):
    vec = np.stack([x, y, z], axis=-1)
    z_axis = np.array([0.0, 0.0, 1.0])
    east = np.cross(z_axis, vec)
    east /= np.linalg.norm(east, axis=-1, keepdims=True) + 1e-8
    north = np.cross(vec, east)
    north /= np.linalg.norm(north, axis=-1, keepdims=True) + 1e-8
    return east, north

def seed_particles():
    initial_positions = []
    ny, nx = grid_shape
    for face in faces:
        x2d, y2d = np.meshgrid(np.arange(nx), np.arange(ny))
        x3d, y3d, z3d = face_to_sphere(face, x2d, y2d)
        grid_coords = np.stack([x3d, y3d, z3d], axis=2)
        indices = np.random.choice(nx * ny, particles_per_face, replace=False)
        for idx in indices:
            j, i = np.unravel_index(idx, (ny, nx))
            initial_positions.append((face, grid_coords[j, i, :].copy()))
    return initial_positions

# ------------------------------------------------------------------------------
# Wind Field Loading and Preprocessing
# ------------------------------------------------------------------------------

def load_dataset_for_tracking(face):
    url_u = f"{base_url}/GEOS_U/u_face_{face}_depth_52_time_0_10269.idx{common_params}"
    url_v = f"{base_url}/GEOS_V/v_face_{face}_depth_52_time_0_10269.idx{common_params}"
    try:
        db_u = ov.LoadDataset(url_u)
        db_v = ov.LoadDataset(url_v)
        return db_u, db_v
    except Exception as e:
        print(f"Failed to load face {face}: {e}")
        return None, None

def preload_wind_fields(face):
    db_u, db_v = load_dataset_for_tracking(face)
    if not db_u or not db_v:
        return None
    u_fields, v_fields = [], []
    for t in timesteps:
        try:
            data_u = db_u.read(time=t, quality=-12, z=[z_level, z_level+1])[0]
            data_v = db_v.read(time=t, quality=-12, z=[z_level, z_level+1])[0]
            u_fields.append(data_u)
            v_fields.append(data_v)
        except Exception as e:
            print(f"Error loading face {face} at time {t}: {e}")
            u_fields.append(None)
            v_fields.append(None)
    return (np.array(u_fields), np.array(v_fields))

def preload_all_wind_fields():
    wind_cache = {}
    for face in faces:
        fields = preload_wind_fields(face)
        if fields is not None:
            wind_cache[face] = fields
    return wind_cache

def precompute_face_grids():
    face_data = {}
    for face in faces:
        ny, nx = grid_shape
        x2d, y2d = np.meshgrid(np.arange(nx), np.arange(ny))
        x3d, y3d, z3d = face_to_sphere(face, x2d, y2d)
        east, north = compute_tangent_vectors(x3d, y3d, z3d)
        grid_coords = np.stack([x3d, y3d, z3d], axis=2)
        face_data[face] = (grid_coords, east, north)
    return face_data

# ------------------------------------------------------------------------------
# Interpolation Utility
# ------------------------------------------------------------------------------

def interpolate_vector(P, grid_coords, vector_field):
    flat_grid = grid_coords.reshape(-1, 3)
    flat_vectors = vector_field.reshape(-1, 3)
    dists = np.linalg.norm(flat_grid - P, axis=1)
    weights = 1 / (dists + 1e-6)
    weights /= weights.sum()
    interpolated = np.sum(flat_vectors * weights[:, None], axis=0)
    return interpolated

# ------------------------------------------------------------------------------
# Particle Tracking (with interpolated velocities)
# ------------------------------------------------------------------------------

def track_particle(P0, face, face_data, wind_fields):
    trajectory = [P0.copy()]
    P = P0.copy()
    if wind_fields is None:
        return trajectory

    grid_coords_full, east_full, north_full = face_data[face]
    u_fields, v_fields = wind_fields

    for idx, t in enumerate(timesteps):
        data_u = u_fields[idx]
        data_v = v_fields[idx]
        if data_u is None or data_v is None:
            break

        ny, nx = data_u.shape
        if grid_coords_full.shape[:2] != (ny, nx):
            y_indices = np.linspace(0, grid_coords_full.shape[0]-1, ny).astype(int)
            x_indices = np.linspace(0, grid_coords_full.shape[1]-1, nx).astype(int)
            grid_coords = grid_coords_full[np.ix_(y_indices, x_indices)]
            east = east_full[np.ix_(y_indices, x_indices)]
            north = north_full[np.ix_(y_indices, x_indices)]
        else:
            grid_coords = grid_coords_full
            east = east_full
            north = north_full

        u3d = data_u[..., None] * east + data_v[..., None] * north
        vel = interpolate_vector(P, grid_coords, u3d)

        P += 0.01 * vel
        P /= np.linalg.norm(P)
        trajectory.append(P.copy())
    return trajectory

# ------------------------------------------------------------------------------
# Main Driver
# ------------------------------------------------------------------------------

print("Precomputing global face data...")
face_data_global = precompute_face_grids()
print("Loading wind cache...")
wind_cache_global = preload_all_wind_fields()
print("Ready to track.")

def process_particle(args):
    face, P0 = args
    return track_particle(P0, face, face_data_global, wind_cache_global.get(face, None))

def generate_pathlines():
    particles = seed_particles()
    num_cores = -1
    pathlines = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(process_particle)(particle) for particle in tqdm(particles)
    )
    with open(os.path.join(cache_dir, "pathlines.pkl"), "wb") as f:
        pickle.dump(pathlines, f)
    print("Pathlines saved to cache.")
    return pathlines

if __name__ == "__main__":
    generate_pathlines()
