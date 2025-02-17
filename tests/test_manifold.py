from e3nn.o3 import rand_matrix
from neural_surfaces import Manifold
from neural_surfaces.utils import OdedSteinMeshes
from potpourri3d import cotan_laplacian
from torch import cat, float64, ones_like, pi, randn, sqrt, tensor, zeros_like
from torch.linalg import norm
from torch.testing import assert_close
from trimesh.primitives import Sphere
from unittest import main, TestCase


meshes = OdedSteinMeshes()
spot = meshes.spot()

class TestManifold(TestCase):
    def test_meshes(self):
        """Checks that available meshes are boundaryless with Euler characteristic 2"""
        for name in meshes.names:
            _, faces = getattr(meshes, name)()
            m = Manifold(faces)
            
            self.assertTrue(m.is_boundary_halfedge.sum().item() == 0)
            self.assertTrue(m.euler_char == 2)

    def test_embedding_to_halfedge_vectors(self):
        """Checks that halfedge vectors sum to 0 around faces"""
        fs, faces = spot
        m = Manifold(faces)

        e_sums = m.embedding_to_halfedge_vectors(fs)[m.halfedges_to_faces].sum(dim=-2)
        assert_close(e_sums, zeros_like(e_sums))

    def test_metric(self):
        """Checks that discrete metric is nonnegative and satisfies the triangle inequality on each face"""
        fs, faces = spot
        m = Manifold(faces)

        ls = m.embedding_to_metric(fs)
        self.assertTrue((ls > 0).all())
        
        l_ijs = ls[faces]
        l_jks = l_ijs[:, tensor([1, 2, 0])]
        l_kis = l_ijs[:, tensor([2, 0, 1])]
        self.assertTrue((l_ijs + l_jks >= l_kis).all() and (l_jks + l_kis >= l_ijs).all() and (l_kis + l_ijs >= l_jks).all())
        
    def test_angles(self):
        """Checks that angles are nonnegative, sum to pi on each face, and agree when computed from embedding or from discrete metric"""
        fs, faces = spot
        m = Manifold(faces)
        ls = m.embedding_to_metric(fs)

        alphas = m.embedding_to_angles(fs)
        self.assertTrue((alphas >= 0).all())
        alpha_sums = alphas.reshape(m.num_faces, 3).sum(dim=-1)
        assert_close(alpha_sums, pi * ones_like(alpha_sums))

        assert_close(m.embedding_to_angles(fs), m.metric_to_angles(ls))
    
    def test_face_areas(self):
        """Checks that face areas are nonnegative and agree when computed from embedding or from discrete metric"""
        fs, faces = spot
        m = Manifold(faces)
        ls = m.embedding_to_metric(fs)

        As = m.embedding_to_face_areas(fs)
        self.assertTrue((As >= 0).all())

        assert_close(As, m.metric_to_face_areas(ls))

    def test_laplacian(self):
        """Checks Laplacian against potpourri3d Laplacian"""
        fs, faces = spot
        m = Manifold(faces)
        L = m.embedding_to_laplacian(fs).to_dense()
        
        L_comp = -tensor(cotan_laplacian(fs.numpy(), faces.numpy()).todense())
        assert_close(L, L_comp)

    def test_vertex_areas(self):
        """Checks that vertex areas are nonnegative and have same sum as face areas"""
        fs, faces = spot
        m = Manifold(faces)
        face_As = m.embedding_to_face_areas(fs)
        vertex_As = m.face_areas_to_vertex_areas(face_As)

        self.assertTrue((vertex_As > 0.).all())

        assert_close(vertex_As.sum(), face_As.sum())

    def test_mass_matrix(self):
        """Checks that mass matrix is nonnegative, symmetric, and rows sum to vertex areas"""
        fs, faces = spot
        m = Manifold(faces)
        M = m.embedding_to_mass_matrix(fs)
        dense_M = M.to_dense()

        self.assertTrue((dense_M >= 0).all())
        assert_close(dense_M, dense_M.T)

        vertex_As = m.face_areas_to_vertex_areas(m.embedding_to_face_areas(fs))
        assert_close((M @ ones_like(vertex_As)), vertex_As)

    def test_grad_and_div(self):
        """Checks that gradient and divergence computations are adjoint"""
        fs, faces = spot
        m = Manifold(faces)

        f = randn(m.num_vertices, dtype=float64)
        G = randn(m.num_faces, 3)
        Ns = m.embedding_to_face_normals(fs)
        G -= (Ns * G).sum(dim=-1, keepdims=True) * Ns

        A = m.embedding_to_face_areas(fs)
        face_inner_product = (A * (m.embedding_and_vertex_values_to_face_grads(fs, f) * G).sum(dim=-1)).sum()

        M = m.face_areas_to_vertex_areas(A)
        vertex_inner_product = -(f * (M * m.embedding_and_face_vectors_to_vertex_divs(fs, G, use_diag_mass=True))).sum()

        assert_close(face_inner_product, vertex_inner_product)

    def test_flip_halfedge(self):
        faces = tensor([[0, 1, 2], [3, 2, 1]])
        m = Manifold(faces)
        m.flip_halfedge(1)

        self.assertTrue((m.faces == tensor([[1, 3, 0], [2, 0, 3]])).all())
        self.assertTrue((m.tails_to_halfedges == tensor([0, 3, 2, 3, 0, 1])).all())
        self.assertTrue((m.tips_to_halfedges == tensor([1, 0, 0, 2, 3, 3])).all())
        assert_close(m.halfedges_to_tails.to_dense(), tensor([[1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]], dtype=float64))
        self.assertTrue((m.halfedges_to_faces == tensor([[5, 1, 0], [2, 4, 3]])).all())
        self.assertTrue((m.faces_to_halfedges == tensor([[5, 1, 0, 2, 4, 3]])).all())
        self.assertTrue((m.halfedges_to_twins == tensor([-1, 4, -1, -1, 1, -1])).all())
        self.assertTrue((m.is_boundary_halfedge == tensor([True, False, True, True, False, True])).all())

    def test_locator(self):
        sphere = Sphere()
        sphere_fs = tensor(sphere.vertices)
        m = Manifold(tensor(sphere.faces))

        query = randn(10, 5, 3, dtype=float64)
        query /= norm(query, dim=-1, keepdims=True)
        face_idxs, barys = m.sphere_embedding_to_locator(sphere_fs)(query)
        recon_query = (sphere_fs[m.faces[face_idxs]] * barys.unsqueeze(-1)).sum(dim=-2)
        recon_query /= norm(recon_query, dim=-1, keepdims=True)
        assert_close(recon_query, query)

    def test_arap(self):
        fs, faces = spot
        m = Manifold(faces)
        R = rand_matrix(dtype=float64)
        R_fs = fs @ R.T
        sigmas = m.frames_to_singular_values(m.embedding_to_frames(fs), m.embedding_to_frames(R_fs))
        assert_close(sigmas, ones_like(sigmas))

    def test_conformal_flattening(self):
        fs, faces = spot
        m = Manifold(faces).remove_vertex()
        fs = fs[1:]
        ls = m.embedding_to_metric(fs)
        flat_ls, _ = m.metric_to_flat_metric(ls, 50)
        flat_fs = m.metric_to_spectral_conformal_parametrization(flat_ls, 20)
        flat_fs = cat([flat_fs, zeros_like(flat_fs[:, :1])], dim=-1)
        area_ratio = m.metric_to_face_areas(flat_ls).sum() / m.embedding_to_face_areas(flat_fs).sum()
        flat_fs *= sqrt(area_ratio)
        layout_flat_ls = m.embedding_to_metric(flat_fs)
        assert_close(flat_ls, layout_flat_ls)


if __name__ == '__main__':
    main()
