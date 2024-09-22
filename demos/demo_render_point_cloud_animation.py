from neural_surfaces import Manifold
from neural_surfaces.utils import OdedSteinMeshes, point_cloud_trajectories_and_mesh_to_html, serve_html


fs, faces = OdedSteinMeshes().spot()
m = Manifold(faces)
Ns = m.embedding_to_vertex_normals(fs)
_, _, x_trajs = m.embedding_to_samples(fs, 100 * 5)
x_trajs = x_trajs.reshape(5, -1, 3)

serve_html(point_cloud_trajectories_and_mesh_to_html(x_trajs, 0.01, 0.5, fs, faces, Ns))
