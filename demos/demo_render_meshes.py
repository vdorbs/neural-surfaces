from neural_surfaces.utils import BUNNY_URL, CAT_URL, load_obj_from_url, PENGUIN_URL, render_meshes, SPOT_URL
from torch import atan2, pi, stack
import wandb
from wandb import Html


all_fs = []
all_faces = []
all_uvs = []

for url_row in [[BUNNY_URL, SPOT_URL], [CAT_URL, PENGUIN_URL]]:
    fs_row = []
    faces_row = []
    uvs_row = []
    
    for url in url_row:
        fs, faces = load_obj_from_url(url)
        us = (atan2(fs[:, 2], fs[:, 0]) + pi) / (2 * pi)
        vs = (fs[:, 1] - fs[:, 1].min()) / (fs[:, 1].max() - fs[:, 1].min())
        uvs = stack([us, vs], dim=-1)

        fs_row.append(fs)
        faces_row.append(faces)
        uvs_row.append(uvs)

    all_fs.append(fs_row)
    all_faces.append(faces_row)
    all_uvs.append(uvs_row)

html_str = render_meshes(all_fs, all_faces, all_uvs)

# Log to wandb
wandb.init()
wandb.log(dict(html=Html(html_str)))
wandb.finish()
