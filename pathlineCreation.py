# Creation CODE !!!
import numpy as np
import OpenVisus as ov
import os
import pickle
from tqdm import tqdm
# Use joblib for parallel processing (faster than multiprocessing.Pool)
from joblib import Parallel, delayed

# ------------------------------------------------------------------------------
# Configuration and Setup
# ------------------------------------------------------------------------------

os.environ['VISUS_CACHE'] = "./visus_can_be_deleted"
cache_dir = "pathline_cache"
os.makedirs(cache_dir, exist_ok=True)

# Parameters (customize here)
variable_u, variable_v = 'u', 'v'
faces = range(6)
base_url = "https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/GEOS"
common_params = "?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco"
z_level = 50
timesteps = list(range(0, 1000, 5))   # Custom timespan; adjust as needed
grid_shape = (48, 48)
particles_per_face = 1000          # You may lower this for faster testing

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

def find_nearest_index(P, grid_coords):
    diff = grid_coords - P
    dist2 = np.sum(diff**2, axis=2)
    return np.unravel_index(np.argmin(dist2), dist2.shape)

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
            data_u = db_u.read(time=t, quality=-4, z=[z_level, z_level+1])[0]
            data_v = db_v.read(time=t, quality=-4, z=[z_level, z_level+1])[0]
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
# Particle Tracking (with resolution handling)
# ------------------------------------------------------------------------------

def track_particle(P0, face, face_data, wind_fields):
    trajectory = [P0.copy()]
    P = P0.copy()
    if wind_fields is None:
        return trajectory
    # Get the precomputed grid for this face
    grid_coords_full, east_full, north_full = face_data[face]
    u_fields, v_fields = wind_fields

    for idx, t in enumerate(timesteps):
        data_u = u_fields[idx]
        data_v = v_fields[idx]
        if data_u is None or data_v is None:
            break

        # Check if the resolution of wind data matches the precomputed grid.
        ny, nx = data_u.shape
        if grid_coords_full.shape[:2] != (ny, nx):
            # Re-index the grid to the resolution of the wind data.
            # Using np.linspace to pick indices along each axis.
            y_indices = np.linspace(0, grid_coords_full.shape[0]-1, ny).astype(int)
            x_indices = np.linspace(0, grid_coords_full.shape[1]-1, nx).astype(int)
            # Use np.ix_ to generate a 2D index grid for multi-indexing
            grid_coords = grid_coords_full[np.ix_(y_indices, x_indices)]
            east = east_full[np.ix_(y_indices, x_indices)]
            north = north_full[np.ix_(y_indices, x_indices)]
        else:
            grid_coords = grid_coords_full
            east = east_full
            north = north_full

        # Compute the full 3D velocity vector using the local tangent basis.
        u3d = data_u[..., None] * east + data_v[..., None] * north

        # Find the grid index nearest to the current particle position.
        j, i = find_nearest_index(P, grid_coords)
        # Adjust the particle's position (tuning of step size might help further speed).
        P += 0.01 * u3d[j, i]
        P /= np.linalg.norm(P)
        trajectory.append(P.copy())
    return trajectory

# Precompute globals so that they are shared (read-only) in parallel processes.
face_data_global = precompute_face_grids()
wind_cache_global = preload_all_wind_fields()

def process_particle(args):
    face, P0 = args
    return track_particle(P0, face, face_data_global, wind_cache_global.get(face, None))

def generate_pathlines():
    particles = seed_particles()  # List of (face, initial_position)
    num_cores = -1  # Use all available cores
    # Run in parallel with Joblib; global data is shared read-only.
    pathlines = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(process_particle)(particle) for particle in tqdm(particles)
    )
    with open(os.path.join(cache_dir, "pathlines.pkl"), "wb") as f:
        pickle.dump(pathlines, f)
    print("Pathlines saved to cache.")
    return pathlines

if __name__ == "__main__":
    generate_pathlines()