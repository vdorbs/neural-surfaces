from argparse import ArgumentParser
from neural_surfaces import Manifold
from neural_surfaces.render import Scene
from neural_surfaces.utils import OdedSteinMeshes, serve_html, sparse_solve, write_html
from os import environ
from potpourri3d import read_mesh
from re import match
from torch import device, pi, save, sqrt, stack, tensor
from tqdm import tqdm
import wandb
from wandb import Html


parser = ArgumentParser()
parser.add_argument('--mesh_path')
parser.add_argument('--device', default='cpu')
parser.add_argument('--h', type=float, default=1e-2)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--log_every', type=int, default=1)
parser.add_argument('--y_up', type=bool, default=False)
parser.add_argument('--frame_length', type=int, default=1)
parser.add_argument('--log_to_wandb', type=bool, default=False)
parser.add_argument('--wandb_project')
parser.add_argument('--output')
args = parser.parse_args()

# Set device for cholespy on multi-GPU systems
matching = match(r'(?:cuda:|)([0-9]{1,})', args.device)
if matching is not None:
    environ['CUDA_VISIBLE_DEVICES'] = matching.group(1)
    device = device('cuda:0')
elif args.device == 'cpu':
    device = device('cpu')
else:
    raise NotImplementedError

if args.mesh_path is not None:
    fs, faces = map(tensor, read_mesh(args.mesh_path))
else:
    fs, faces = OdedSteinMeshes().spot()

manifold = Manifold(faces).to(device)
fs = fs.to(device)
fs -= manifold.embedding_to_com(fs)
fs *= sqrt(4 * pi / manifold.embedding_to_face_areas(fs).sum())
L = manifold.embedding_to_laplacian(fs)

if args.log_to_wandb:
    wandb.init(project=args.wandb_project, config=args)

frames = [fs]
if args.log_to_wandb:
    Ns = manifold.embedding_to_vertex_normals(fs).cpu()
    scene = Scene()
    scene.add_mesh(fs.cpu(), faces, Ns, y_up=args.y_up)
    wandb.log(dict(surface=Html(scene.make())), step=0)

for step in tqdm(range(args.num_steps)):
    M = manifold.embedding_to_mass_matrix(fs)
    A = M - args.h * L
    fs = sparse_solve(A, M @ fs)
    fs -= manifold.embedding_to_com(fs)
    fs *= sqrt(4 * pi / manifold.embedding_to_face_areas(fs).sum())

    if (step + 1) % args.log_every == 0:
        frames.append(fs)

        if args.log_to_wandb:
            Ns = manifold.embedding_to_vertex_normals(fs).cpu()
            scene = Scene()
            scene.add_mesh(fs.cpu(), faces, Ns, y_up=args.y_up)
            wandb.log(dict(surface=Html(scene.make())), step=step)

frames = stack(frames)
all_Ns = stack([manifold.embedding_to_vertex_normals(fs).cpu() for fs in frames])
frames = frames.cpu()
scene = Scene(num_frames=len(frames), frame_length=args.frame_length)
scene.add_mesh(frames, faces, all_Ns, y_up=args.y_up, is_animated=True)
scene_html = scene.make()

if args.log_to_wandb:
    wandb.log(dict(animation=Html(scene_html)))
    wandb.finish()

if args.output is not None:
    save(dict(frames=frames, faces=faces), f'{args.output}_data.pt')
    write_html(f'{args.output}_anim.html', scene_html)

serve_html(scene_html, serve_locally=False)
