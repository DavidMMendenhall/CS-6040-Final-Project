import numpy as np
import OpenVisus as ov
import os
import pickle
os.environ['VISUS_CACHE'] = "./visus_can_be_deleted"

base_url = "https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/GEOS"
common_params = "?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco"
dx, dy = 4.0, 4.0
z_level = 50

timeStart = 0
timeEnd = 10269
step = 50

vorticity_data = None
vorticity_db = []
print("Connecting databases")
for face in range(6):
    u_url = f"{base_url}/GEOS_U/u_face_{face}_depth_52_time_0_10269.idx{common_params}"
    v_url = f"{base_url}/GEOS_V/v_face_{face}_depth_52_time_0_10269.idx{common_params}"
    try:
        db_u = ov.LoadDataset(u_url)
        db_v = ov.LoadDataset(v_url)
        vorticity_db.append((db_u, db_v))
        
    except Exception as e:
        print(f"Face {face} error: {e}")
print("loading data_u...")
data_u_raw = [[vorticity_db[face][0].read(time=time_step, quality=-6, z=[z_level, z_level + 1]) for face in range(6)] for time_step in range(timeStart, timeEnd, step)]
print("loading data_v...")
data_v_raw = [[vorticity_db[face][1].read(time=time_step, quality=-6, z=[z_level, z_level + 1]) for face in range(6)] for time_step in range(timeStart, timeEnd, step)]
for time_step in range(len(data_u_raw)):
    print(f"time step: {time_step}")
    vorticities = [None] * 6
    for face in range(6):
        data_u = data_u_raw[time_step][face]
        data_v = data_v_raw[time_step][face]
        u, v = data_u[0, :, :], data_v[0, :, :]
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        vort = dvdx - dudy

        # Orientation corrections per face
        if face == 0:
            vort = np.rot90(vort, k=1)
        elif face == 1:
            vort = np.rot90(vort, k=1)
        elif face == 3:
            vort = np.rot90(vort, k=2)
        elif face == 4:
            vort = np.rot90(vort, k=2)
        elif face == 5:
            vort = np.rot90(vort, k=1)

        vorticities[face] = vort
    
    all_vorticities = np.concatenate([vort.flatten() for vort in vorticities if vort is not None])
    if vorticity_data is None:
        vorticity_data = all_vorticities
    else:
        vorticity_data = np.concatenate((vorticity_data, all_vorticities))

vorticity_data.tofile(f"./vort_data/vorticities_z{z_level}_{timeStart}-{timeEnd}-{step}_{data_u[0][0].shape[1]}")
    

# filename = f"all_vorticities_z{z_level}_timestamp{timestep}.pkl"
# with open(filename, "wb") as f:
#     pickle.dump(all_vorticities, f)
# print(f"Saved {filename} with shape {all_vorticities.shape}")
# filename = f"all_vorticities_z{z_level}_timestamp{timestep}.pkl"

# # Save vorticities as pickle file
# vorticity_dict = {face_idx: vort for face_idx, vort in enumerate(vorticities) if vort is not None}
# output_filename = f"all_vorticities_z{z_level}_timestamp{timestep}.pkl"
# with open(output_filename, "wb") as f:
#     pickle.dump(vorticity_dict, f)
# print(f"Vorticities saved to {output_filename}")`

# # Reading
# import pickle

# filename = f"all_vorticities_z{z_level}_timestamp{timestep}.pkl"

# with open(filename, "rb") as f:
#     vorticity_dict = pickle.load(f)

# for face_idx, vorticity_array in vorticity_dict.items():
#     print(f"Face {face_idx} — vorticity values:")
#     print(vorticity_array)`