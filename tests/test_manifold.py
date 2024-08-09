from neural_surfaces import Manifold
from neural_surfaces.utils import load_obj_from_url
from torch import ones_like, pi, tensor
from torch.testing import assert_close
from unittest import TestCase


spot_url = 'https://raw.githubusercontent.com/odedstein/meshes/master/objects/spot/spot_low_resolution.obj'

class TestManifold(TestCase):
    def test_metric(self):
        """Checks that discrete metric is nonnegative and satisfies the triangle inequality on each face"""
        fs, faces = load_obj_from_url(spot_url)
        m = Manifold(faces)

        ls = m.embedding_to_metric(fs)
        self.assertTrue((ls > 0).all())
        
        l_ijs = m.group_halfedge_lengths_by_face(ls)
        l_jks = l_ijs[:, tensor([1, 2, 0])]
        l_kis = l_ijs[:, tensor([2, 0, 1])]
        self.assertTrue((l_ijs + l_jks >= l_kis).all() and (l_jks + l_kis >= l_ijs).all() and (l_kis + l_ijs >= l_jks).all())
        
    def test_angles(self):
        """Checks that angles are nonnegative, sum to pi on each face, and agree when computed from embedding or from discrete metric"""
        fs, faces = load_obj_from_url(spot_url)
        m = Manifold(faces)
        ls = m.embedding_to_metric(fs)

        alphas = m.embedding_to_angles(fs)
        self.assertTrue((alphas >= 0).all())
        alpha_sums = alphas.reshape(m.num_faces, 3).sum(dim=-1)
        assert_close(alpha_sums, pi * ones_like(alpha_sums))

        assert_close(m.embedding_to_angles(fs), m.metric_to_angles(ls))
    
    def test_face_areas(self):
        """Checks that face areas are nonnegative and agree when computed from embedding or from discrete metric"""
        fs, faces = load_obj_from_url(spot_url)
        m = Manifold(faces)
        ls = m.embedding_to_metric(fs)

        As = m.embedding_to_face_areas(fs)
        self.assertTrue((As >= 0).all())

        assert_close(As, m.metric_to_face_areas(ls))
