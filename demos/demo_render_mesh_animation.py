from neural_surfaces import Manifold
from neural_surfaces.utils import OdedSteinMeshes, mesh_trajectories_to_html, serve_html
from torch import float64, linspace, stack, zeros

fs, faces = OdedSteinMeshes().spot()
m = Manifold(faces)
Ns = m.embedding_to_vertex_normals(fs)

ts = linspace(0, 1, 100 + 1, dtype=float64)
fs_traj = stack([fs + t for t in ts])
Ns_traj = stack([Ns for _ in ts])

uvs = zeros(m.num_vertices, 2, dtype=float64)

serve_html(mesh_trajectories_to_html(fs_traj, faces, Ns_traj, uvs))
