from neural_surfaces import Manifold
from neural_surfaces.render import MultiScene
from neural_surfaces.utils import OdedSteinMeshes, serve_html


names = ['spot', 'bunny', 'penguin']
scene = MultiScene(1, 3)
meshes = OdedSteinMeshes(is_high_res=True)

for i, name in enumerate(names):
    fs, faces = getattr(meshes, names[i])()
    manifold = Manifold(faces)
    sphere_fs = manifold.embedding_to_sphere_embedding(fs, flatten_iters=50, layout_iters=20, center_iters=50, verbose=True)
    scene.add_mesh(0, i, *manifold.sphere_embedding_and_embedding_to_plot_data(sphere_fs, fs, y_up=True))

serve_html(scene.make())
