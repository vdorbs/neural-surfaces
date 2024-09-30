from matplotlib.pyplot import figure, show
from neural_surfaces import Manifold
from neural_surfaces.utils import OdedSteinMeshes


fs, faces = OdedSteinMeshes(is_high_res=True).spot()
manifold = Manifold(faces).remove_vertex()
fs = fs[1:]

uvs = manifold.embedding_to_tutte_parametrization(fs)

ax = figure(figsize=(6, 6)).add_subplot(1, 1, 1)
ax.triplot(*uvs.T, triangles=manifold.faces, linewidth=0.5)
ax.axis('equal')
show()
