from argparse import ArgumentParser
from neural_surfaces import Manifold
from polyscope import init, register_point_cloud, register_surface_mesh, set_user_callback, show
from polyscope.imgui import Button, InputFloat, InputInt
from torch.autograd import grad
from torch import cos, diag, eye, float64, pi, randn, sinc, stack, tensor
from torch.linalg import norm
from tqdm import tqdm
from trimesh.primitives import Sphere


parser = ArgumentParser()
parser.add_argument('--num_points', type=int, required=True)
parser.add_argument('--num_subdivisions', type=int, default=4)
parser.add_argument('--diff_coeff', type=float, default=1.)
parser.add_argument('--init', choices=['uniform', 'clustered'], default='uniform')
parser.add_argument('--loss', choices=['inv_dists', 'inv_squared_dists', 'neg_dists'], default='inv_squared_dists')
args = parser.parse_args()

sphere = Sphere(subdivisions=args.num_subdivisions)
fs = tensor(sphere.vertices)
faces = tensor(sphere.faces)
manifold = Manifold(faces)
locator = manifold.sphere_embedding_to_locator(fs)
solver = manifold.embedding_to_heat_method_solver(fs, use_diag_mass=True, diff_coeff=args.diff_coeff)

def compute_grads(samples):
    face_idxs, barys = locator(samples)
    dists = solver(face_idxs, False, barys).T

    samples_with_grad = samples.clone().requires_grad_(True)
    samples_with_grad = samples_with_grad / norm(samples_with_grad, dim=-1, keepdims=True)
    face_idxs_with_grad, barys_with_grad = locator(samples_with_grad)
    pairwise_dists = stack([(barys_with_grad * dist_row[faces[face_idxs_with_grad]]).sum(dim=-1) for dist_row in dists])
    pairwise_dists = pairwise_dists - diag(diag(pairwise_dists))

    if args.loss == 'inv_dists':
        losses = 1 / (pairwise_dists + eye(len(samples), dtype=float64)) - eye(len(samples), dtype=float64)
    elif args.loss == 'inv_squared_dists':
        losses = 1 / ((pairwise_dists ** 2) + eye(len(samples), dtype=float64)) - eye(len(samples), dtype=float64)
    elif args.loss == 'neg_dists':
        losses = -pairwise_dists

    loss = losses.sum() / (len(samples) * (len(samples) - 1))
    grads, = grad(loss, samples_with_grad)
    return loss, grads

if args.init == 'uniform':
    _, _, samples = manifold.embedding_to_samples(fs, args.num_points)
    samples = samples / norm(samples, dim=-1, keepdims=True)
elif args.init == 'clustered':
    samples = tensor([[0, 0, 1]], dtype=float64) + randn(args.num_points, 3, dtype=float64)
    samples = samples / norm(samples, dim=-1, keepdims=True)
else:
    raise NotImplementedError

loss, grads = compute_grads(samples)

init()
mesh = register_surface_mesh('sphere', fs.numpy(), faces.numpy())
point_cloud = register_point_cloud('points', samples.numpy())
vector_scale = 0.1
point_cloud.add_vector_quantity('neg_grads', -vector_scale * grads.numpy(), vectortype='ambient', enabled=True)

num_steps = 1
step_size = 0.5
max_grad_size = pi / 4
def callback():
    global samples, loss, grads, num_steps, step_size, max_grad_size, vector_scale

    _, num_steps = InputInt('num_steps', num_steps, step=1, step_fast=10)
    _, step_size = InputFloat('step_size', step_size)
    _, max_grad_size = InputFloat('max_grad_size', max_grad_size)
    _, vector_scale = InputFloat('vector_scale', vector_scale)

    if Button('Advance'):
        pbar = tqdm(range(num_steps))
        for _ in pbar:
            curr_step_size = step_size
            grad_size = norm(grads, dim=-1).max()
            num_divs = 0
            while curr_step_size * grad_size > max_grad_size:
                curr_step_size /= 2
                num_divs += 1

            pbar.set_description(f'loss={loss} num_divs={num_divs}')

            vs = -curr_step_size * grads
            norm_vs = norm(vs, dim=-1, keepdims=True)
            samples = cos(norm_vs) * samples + sinc(norm_vs / pi) * vs
            samples = samples / norm(samples, dim=-1, keepdims=True)
            
            loss, grads = compute_grads(samples)
            point_cloud.update_point_positions(samples.numpy())
            point_cloud.add_vector_quantity('neg_grads', -vector_scale * grads.numpy(), vectortype='ambient')

set_user_callback(callback)
show()
