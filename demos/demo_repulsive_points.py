from argparse import ArgumentParser
from neural_surfaces import Manifold
from neural_surfaces.utils import point_cloud_trajectories_and_mesh_to_html, sphere_exp
from potpourri3d import read_mesh
from torch.autograd import grad
from torch import device, diag, eye, float64, inf, pi, randn, sqrt, stack, tensor
from torch.linalg import norm
from tqdm import tqdm
import wandb
from wandb import Html


parser = ArgumentParser()
parser.add_argument('--mesh_path', required=True)
parser.add_argument('--param_path', required=True)
parser.add_argument('--device', default='cpu')
parser.add_argument('--diff_coeff', type=float, default=1.)
parser.add_argument('--num_points', type=int, required=True)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--step_size', type=float, default=0.1)
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--plot_every', type=int, default=10)
parser.add_argument('--y_up', type=bool, default=False)
args = parser.parse_args()

device = device(args.device)

fs, faces = map(tensor, read_mesh(args.mesh_path))
manifold = Manifold(faces)
fs -= manifold.embedding_to_com(fs)
fs *= sqrt(4 * pi / manifold.embedding_to_face_areas(fs).sum())
Ns = manifold.embedding_to_vertex_normals(fs)

sphere_fs, _ = map(tensor, read_mesh(args.param_path))

sphere_fs = sphere_fs.to(device)
fs = fs.to(device)
manifold = manifold.to(device)

locator = manifold.sphere_embedding_to_locator(sphere_fs)
solver = manifold.embedding_to_heat_method_solver(fs, use_diag_mass=True, diff_coeff=args.diff_coeff)

sphere_samples = randn(args.num_points, 3, dtype=float64, device=device)
sphere_samples /= norm(sphere_samples, dim=-1, keepdims=True)

I = eye(args.num_points, dtype=float64, device=device)
pbar = tqdm(range(args.num_steps))
frames = []

wandb.init()

for step in pbar:
    face_idxs, barys = locator(sphere_samples)
    all_dists = solver(face_idxs, False, barys).T

    sphere_samples_with_grad = sphere_samples.clone().requires_grad_(True)
    sphere_samples_with_grad = sphere_samples_with_grad / norm(sphere_samples_with_grad, dim=-1, keepdims=True)
    face_idxs_with_grad, barys_with_grad = locator(sphere_samples_with_grad)
    pairwise_dists = stack([(barys_with_grad * dist_row[manifold.faces[face_idxs_with_grad]]).sum(dim=-1) for dist_row in all_dists])
    pairwise_dists = pairwise_dists - diag(diag(pairwise_dists))

    losses = 1 / (pairwise_dists ** 2 + I) - I
    loss = losses.sum() / (args.num_points * (args.num_points - 1))
    grads, = grad(loss, sphere_samples_with_grad)

    wandb.log(dict(loss=loss), step=step)

    max_grad_norm = norm(grads, dim=-1).max()
    curr_step = args.step_size
    power = 0
    while curr_step * max_grad_norm > args.max_grad_norm:
        curr_step /= 2
        power -= 1

    pbar.set_description(f'loss={loss.item()} power={power}')

    sphere_samples = sphere_exp(sphere_samples, -curr_step * grads)

    if (step + 1) % args.plot_every == 0:
        face_idxs, barys = locator(sphere_samples)
        samples = (barys.unsqueeze(-1) * fs[faces[face_idxs]]).sum(dim=-2)
        frames.append(samples.clone().cpu())
        
frames = stack(frames)
wandb.log(dict(anim=Html(point_cloud_trajectories_and_mesh_to_html(frames, 0.05, 1., fs.cpu(), manifold.faces.cpu(), Ns, args.y_up))), step=args.num_steps)

wandb.finish()