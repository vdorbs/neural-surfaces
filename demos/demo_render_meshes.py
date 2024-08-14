from neural_surfaces.utils import meshes_to_html, OdedSteinMeshes, serve_html
from torch import atan2, pi, stack

all_fs = []
all_faces = []
all_uvs = []

meshes = OdedSteinMeshes()
for data_row in [[meshes.bunny(), meshes.spot()], [meshes.cat(), meshes.penguin()]]:
    fs_row = []
    faces_row = []
    uvs_row = []
    
    for fs, faces in data_row:
        us = (atan2(fs[:, 2], fs[:, 0]) + pi) / (2 * pi)
        vs = (fs[:, 1] - fs[:, 1].min()) / (fs[:, 1].max() - fs[:, 1].min())
        uvs = stack([us, vs], dim=-1)

        fs_row.append(fs)
        faces_row.append(faces)
        uvs_row.append(uvs)

    all_fs.append(fs_row)
    all_faces.append(faces_row)
    all_uvs.append(uvs_row)

serve_html(meshes_to_html(all_fs, all_faces, all_uvs))
