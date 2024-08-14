from neural_surfaces import Manifold
from neural_surfaces.utils import load_obj_from_url, SPOT_URL
from polyscope import init, register_point_cloud, register_surface_mesh, show
from torch import pi, sqrt, tensor


fs, faces = load_obj_from_url(SPOT_URL)
m = Manifold(faces)
fs -= m.embedding_to_com(fs)
fs *= sqrt(4 * pi / m.embedding_to_face_areas(fs).sum())
heat_method_solver = m.embedding_to_heat_method_solver(fs, use_diag_mass=True, diff_coeff=100)

source_idxs = tensor([0, 200, 400])
all_dists = heat_method_solver(source_idxs)

init()

mesh = register_surface_mesh('mesh', fs.numpy(), faces.numpy())
for source_idx, dists in zip(source_idxs, all_dists.T):
    mesh.add_scalar_quantity(f'dist_{source_idx}', -dists, cmap='turbo', isolines_enabled=True, enabled=True)

register_point_cloud('sources', fs[source_idxs].numpy())

show()
