from argparse import ArgumentParser
from natsort import natsorted
from neural_surfaces import Manifold
from neural_surfaces.surface_models import SphericalSurfaceModel
from neural_surfaces.utils import meshes_to_html, serve_html
from os import environ, listdir
from potpourri3d import read_mesh
from torch import device, float64, log10, no_grad, pi, save, sqrt, tensor, zeros
from torch.cuda import set_device
from torch.distributed import all_gather, all_gather_into_tensor, destroy_process_group, init_process_group
from torch.linalg import norm
from torch.multiprocessing import spawn
from torch.optim import Adam
from tqdm import tqdm
import wandb
from wandb import Html


parser = ArgumentParser()
parser.add_argument('--dataset_path', required=True)
parser.add_argument('--output_path')
parser.add_argument('--surfaces', type=int, nargs='*')
parser.add_argument('--devices', type=int, nargs='*')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--log', type=bool, default=False)
args = parser.parse_args()

names = [filename[:-4] for filename in listdir(f'{args.dataset_path}/params') if filename[-4:] == '.obj']
names = natsorted(names)
if args.surfaces is not None:
    names = [names[i] for i in args.surfaces]

if args.devices is None:
    devices = [device(i) for i in range(len(names))]
else:
    devices = [device(i) for i in args.devices]

all_models = [] 
all_optimizers = [] 
all_manifolds = []
all_sphere_fs = [] 
all_fs = []
all_Ns = []
for name in names:
    # Read mesh and parametrization, center and normalize mesh (area of sphere with COM at origin), and compute normals
    fs, faces = map(tensor, read_mesh(f'{args.dataset_path}/meshes/{name}.obj'))
    sphere_fs = tensor(read_mesh(f'{args.dataset_path}/params/{name}.obj')[0])
    manifold = Manifold(faces)
    fs -= manifold.embedding_to_com(fs)
    fs *= sqrt(4 * pi / manifold.embedding_to_face_areas(fs).sum())
    Ns = manifold.embedding_to_face_normals(fs)

    all_manifolds.append(manifold)
    all_sphere_fs.append(sphere_fs)
    all_fs.append(fs)
    all_Ns.append(Ns)

pbar = tqdm(total=args.num_epochs)

def train(rank):
    # Need this otherwise extra process starts on default GPU
    set_device(devices[rank])

    if rank == 0 and args.log:
        wandb.init(config=args)

    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = '12345'
    init_process_group(backend='nccl', rank=rank, world_size=len(devices))

    # Initialize model and optimizer
    model = SphericalSurfaceModel(10, 0.1, 128, 7).to(devices[rank])
    optimizer = Adam(model.parameters())

    # Get training data
    manifold = all_manifolds[rank].to(devices[rank])
    sphere_fs = all_sphere_fs[rank].to(devices[rank])
    fs = all_fs[rank].to(devices[rank])
    Ns = all_Ns[rank].to(devices[rank])

    for _ in range(args.num_epochs):
        # Draw uniform samples from mesh
        face_idxs, barys, target_fs = manifold.embedding_to_samples(fs, 4096)
        target_Ns = Ns[face_idxs]

        # Corresponding samples on sphere
        sample_sphere_fs = (sphere_fs[manifold.faces[face_idxs]] * barys.unsqueeze(-1)).sum(dim=-2)
        sample_sphere_fs = sample_sphere_fs / norm(sample_sphere_fs, dim=-1, keepdims=True)

        # Model predictions
        recon_fs = sample_sphere_fs + model(sample_sphere_fs)
        recon_Ns = model.normal(sample_sphere_fs)

        recon_loss = (norm(recon_fs - target_fs, dim=-1) ** 2).mean()
        normal_loss = (1 - (recon_Ns * target_Ns).sum(dim=-1)).mean()
        loss = recon_loss + 1e-2 * normal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        with no_grad(): recon_fs = sphere_fs + model(sphere_fs)
        recon_loss = (norm(recon_fs - fs, dim=-1) ** 2).mean()

        # Gather losses across devices
        gathered_losses = zeros(len(devices), dtype=float64, device=devices[rank])
        all_gather_into_tensor(gathered_losses, recon_loss)

        if rank == 0:
            pbar.update()
            pbar.set_description(f'loss={gathered_losses.mean().item()}')

            if args.log:
                wandb.log({f'{name}/log_loss':log10(loss) for name, loss in zip(names, gathered_losses)})

    if args.output_path is not None:
        save(model.state_dict, f'{args.output_path}/{names[rank]}.pt')

    # Gather reconstructions across devices
    all_recon_fs = [zeros(*fs.shape, dtype=fs.dtype, device=devices[rank]) for fs in all_fs]
    all_gather(all_recon_fs, recon_fs)

    # No more gathers needed
    destroy_process_group()

    # Render reconstructions
    if rank == 0:
        all_render_fs = [recon_fs.cpu() for recon_fs in all_recon_fs]
        all_render_faces = [manifold.faces.cpu() for manifold in all_manifolds]
        all_render_uvs = [zeros(manifold.num_vertices, 2) for manifold in all_manifolds]

        if args.log:
            for name, render_fs, render_faces, render_uvs in zip(names, all_render_fs, all_render_faces, all_render_uvs):
                wandb.log({f'{name}/recon': Html(meshes_to_html([[render_fs]], [[render_faces]], [[render_uvs]], mode='none'))})
                
            wandb.finish()

        else:
            serve_html(meshes_to_html([all_render_fs], [all_render_faces], [all_render_uvs], mode='none'), serve_locally=False)

if __name__ == '__main__':
    spawn(train, nprocs=len(devices), join=True)
