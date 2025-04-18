import numpy as np
import os
import OpenVisus as ov
import json
from cubeMarch import cube_march

os.environ['VISUS_CACHE']= "./visus_can_be_deleted"

variable = "CO2"
data_urls = [ f"https://maritime.sealstorage.io/api/v0/s3/utah/nasa/dyamond/GEOS/GEOS_{variable.upper()}/{variable.lower()}_face_{face}_depth_52_time_0_10269.idx?access_key=any&secret_key=any&endpoint_url=https://maritime.sealstorage.io/api/v0/s3&cached=arco" for face in range(6)]
db_faces = [ov.LoadDataset(data_url) for data_url in data_urls]

res = -12
startTime = 0
endTime = 10269
# endTime = 100
step = 50
iso_value = 0.000384

meshPoints = None
meshFaces = None
pointOffsets = []
faceOffsets = []

face_offset = 0
point_offset = 0
for time_step in range(startTime, endTime, step):
    for face in range(6):
        print(f"Time step {time_step}, face: {face}")
        mesh = cube_march(db_faces[face].read(time=time_step, quality=res), iso_value)
        mesh[1][::4] = 3 # fix a bug from the compressor
        if meshPoints is None:
            meshPoints = mesh[0]
            meshFaces = mesh[1]
        else:
            meshPoints = np.concat((meshPoints, mesh[0]), axis=0)
            meshFaces = np.concat((meshFaces, mesh[1]), axis=0)
        pointOffsets.append(point_offset)
        faceOffsets.append(face_offset)
        face_offset += mesh[1].shape[0]
        point_offset += mesh[0].shape[0]

meshName = f"{variable}_{startTime}-{endTime}-{step}_{res}"
meshPoints.tofile(f"{meshName}_mesh")
meshFaces.tofile(f"{meshName}_faces")

with open(f"./{meshName}_meta.json", "w") as file:
    json.dump({
        "face_offsets": faceOffsets,
        "mesh_offsets": pointOffsets,
    }, file)