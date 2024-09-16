from argparse import ArgumentParser
from neural_surfaces import Manifold
from neural_surfaces.utils import meshes_to_html, sparse_solve
from potpourri3d import read_mesh
from torch import arcsin, atan2, cos, float64, inf, ones, pi, sinc, sqrt, stack, tensor
from torch.autograd import grad
from torch.linalg import norm
from torch.sparse import spdiags
from tqdm import tqdm
import wandb
from wandb import Html


parser = ArgumentParser()
parser.add_argument('--mesh_path', required=True)
parser.add_argument('--param_path', required=True)
parser.add_argument('--output_path')
parser.add_argument('--y_up', type=bool, default=False)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--step_size', type=float, default=1e-3)
parser.add_argument('--plot_every', type=int, default=10)
args = parser.parse_args()

fs, faces = map(tensor, read_mesh(args.mesh_path))
manifold = Manifold(faces)
fs -= manifold.embedding_to_com(fs)
fs *= sqrt(4 * pi / manifold.embedding_to_face_areas(fs).sum())
As = manifold.embedding_to_face_areas(fs)
frames = manifold.embedding_to_frames(fs)
Ns = manifold.embedding_to_vertex_normals(fs)

sphere_fs, _ = map(tensor, read_mesh(args.param_path))

def plot(sphere_fs):
    all_fs = [[fs]]
    all_faces = [[manifold.faces]]
    all_Ns = [[Ns]]
    
    if args.y_up:
        us = 1 - (atan2(sphere_fs[:, 0], sphere_fs[:, 2]) + pi) / (2 * pi)
        vs = (arcsin(sphere_fs[:, 1]) + pi / 2) / pi
    else:
        us = 1 - (atan2(sphere_fs[:, 1], sphere_fs[:, 0]) + pi) / (2 * pi)
        vs = (arcsin(sphere_fs[:, 2]) + pi / 2) / pi

    uvs = stack([us, vs], dim=-1)
    all_uvs = [[uvs]]
    
    html_str = meshes_to_html(all_fs, all_faces, all_Ns, all_uvs, y_up=args.y_up, mode='checkerboard')
    return Html(html_str)

wandb.init()
wandb.log(dict(param=plot(sphere_fs)), step=0)

pbar = tqdm(range(args.num_steps))
for step in pbar:
    sphere_fs_with_grad = sphere_fs.clone().requires_grad_(True)
    sphere_frames = manifold.embedding_to_frames(sphere_fs_with_grad)
    sigmas = manifold.frames_to_singular_values(frames, sphere_frames)
    squared_sigmas = sigmas ** 2
    sym_dir_energy = (As * (squared_sigmas + 1 / squared_sigmas).sum(dim=-1)).sum()
    grads, = grad(sym_dir_energy, sphere_fs_with_grad)

    L = manifold.embedding_to_laplacian(sphere_fs)
    L_eps = -L + 1e-4 * spdiags(ones(manifold.num_vertices, dtype=float64), tensor(0), shape=(manifold.num_vertices, manifold.num_vertices))
    grads = sparse_solve(L_eps.coalesce(), grads)
    
    curr_step_size = args.step_size
    first_divisions = 0
    has_flips = True
    while has_flips:
        vs = -curr_step_size * grads
        norm_vs = norm(vs, dim=-1, keepdims=True)
        next_sphere_fs = cos(norm_vs) * sphere_fs + sinc(norm_vs / pi) * vs
        next_sphere_fs = next_sphere_fs / norm(next_sphere_fs, dim=-1, keepdims=True)
        next_sphere_Ns = manifold.embedding_to_face_normals(next_sphere_fs)
        has_flips = ((next_sphere_fs[manifold.faces[:, 0]] * next_sphere_Ns).sum() < 0).any()
        
        curr_step_size /= 2
        first_divisions += 1

    curr_step_size *= 2
    first_divisions -= 1

    second_divisions = 0
    next_sym_dir_energy = inf
    while next_sym_dir_energy > sym_dir_energy:
        vs = -curr_step_size * grads
        norm_vs = norm(vs, dim=-1, keepdims=True)
        next_sphere_fs = cos(norm_vs) * sphere_fs + sinc(norm_vs / pi) * vs
        next_sphere_fs = next_sphere_fs / norm(next_sphere_fs, dim=-1, keepdims=True)
        next_sphere_frames = manifold.embedding_to_frames(next_sphere_fs)
        next_sigmas = manifold.frames_to_singular_values(frames, next_sphere_frames)
        next_squared_sigmas = next_sigmas ** 2
        next_sym_dir_energy = (As * (next_squared_sigmas + 1 / next_squared_sigmas).sum(dim=-1)).sum()

        curr_step_size /= 2
        second_divisions += 1

    curr_step_size *= 2
    second_divisions -= 1

    pbar.set_description(f'first_divs={first_divisions} second_divs={second_divisions} energy={sym_dir_energy.item()}')

    sphere_fs = next_sphere_fs

    log_dict = dict(loss=sym_dir_energy)

    if (step + 1) % args.plot_every == 0:
        log_dict['param'] = plot(sphere_fs)

    wandb.log(log_dict, step + 1)

wandb.finish()
