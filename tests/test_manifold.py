from neural_surfaces import Manifold
from neural_surfaces.utils import load_obj_from_url, SPOT_URL
from potpourri3d import cotan_laplacian
from torch import float64, ones_like, pi, randn, tensor, zeros_like
from torch.testing import assert_close
from unittest import TestCase


class TestManifold(TestCase):
    def test_embedding_to_halfedge_vectors(self):
        """Checks that halfedge vectors sum to 0 around faces"""
        fs, faces = load_obj_from_url(SPOT_URL)
        m = Manifold(faces)

        e_sums = m.embedding_to_halfedge_vectors(fs)[m.halfedges_to_faces].sum(dim=-2)
        assert_close(e_sums, zeros_like(e_sums))

    def test_metric(self):
        """Checks that discrete metric is nonnegative and satisfies the triangle inequality on each face"""
        fs, faces = load_obj_from_url(SPOT_URL)
        m = Manifold(faces)

        ls = m.embedding_to_metric(fs)
        self.assertTrue((ls > 0).all())
        
        l_ijs = ls[faces]
        l_jks = l_ijs[:, tensor([1, 2, 0])]
        l_kis = l_ijs[:, tensor([2, 0, 1])]
        self.assertTrue((l_ijs + l_jks >= l_kis).all() and (l_jks + l_kis >= l_ijs).all() and (l_kis + l_ijs >= l_jks).all())
        
    def test_angles(self):
        """Checks that angles are nonnegative, sum to pi on each face, and agree when computed from embedding or from discrete metric"""
        fs, faces = load_obj_from_url(SPOT_URL)
        m = Manifold(faces)
        ls = m.embedding_to_metric(fs)

        alphas = m.embedding_to_angles(fs)
        self.assertTrue((alphas >= 0).all())
        alpha_sums = alphas.reshape(m.num_faces, 3).sum(dim=-1)
        assert_close(alpha_sums, pi * ones_like(alpha_sums))

        assert_close(m.embedding_to_angles(fs), m.metric_to_angles(ls))
    
    def test_face_areas(self):
        """Checks that face areas are nonnegative and agree when computed from embedding or from discrete metric"""
        fs, faces = load_obj_from_url(SPOT_URL)
        m = Manifold(faces)
        ls = m.embedding_to_metric(fs)

        As = m.embedding_to_face_areas(fs)
        self.assertTrue((As >= 0).all())

        assert_close(As, m.metric_to_face_areas(ls))

    def test_laplacian(self):
        """Checks Laplacian against potpourri3d Laplacian"""
        fs, faces = load_obj_from_url(SPOT_URL)
        m = Manifold(faces)
        L = m.embedding_to_laplacian(fs).to_dense()
        
        L_comp = -tensor(cotan_laplacian(fs.numpy(), faces.numpy()).todense())
        assert_close(L, L_comp)

    def test_vertex_areas(self):
        """Checks that vertex areas are nonnegative and have same sum as face areas"""
        fs, faces = load_obj_from_url(SPOT_URL)
        m = Manifold(faces)
        face_As = m.embedding_to_face_areas(fs)
        vertex_As = m.face_areas_to_vertex_areas(face_As)

        self.assertTrue((vertex_As > 0.).all())

        assert_close(vertex_As.sum(), face_As.sum())

    def test_mass_matrix(self):
        """Checks that mass matrix is nonnegative, symmetric, and rows sum to vertex areas"""
        fs, faces = load_obj_from_url(SPOT_URL)
        m = Manifold(faces)
        M = m.embedding_to_mass_matrix(fs)
        dense_M = M.to_dense()

        self.assertTrue((dense_M >= 0).all())
        assert_close(dense_M, dense_M.T)

        vertex_As = m.face_areas_to_vertex_areas(m.embedding_to_face_areas(fs))
        assert_close((M @ ones_like(vertex_As)), vertex_As)

    def test_grad_and_div(self):
        """Checks that gradient and divergence computations are adjoint"""
        fs, faces = load_obj_from_url(SPOT_URL)
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
        