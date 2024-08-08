from torch import arange, arccos, maximum, minimum, sqrt, Tensor, tensor
from torch.linalg import cross, norm
from torch.nn import Module


class Manifold(Module):
    """Stores topological data for a manifold triangle mesh. Computes objects for geometry processing when geometric data (vertex positions, edge lengths, etc.) are provided."""
    
    def __init__(self, faces: Tensor):
        """
        Args:
            faces (Tensor): num_faces * 3 list of vertices per face
        """
        Module.__init__(self)
        self.register_buffer('faces', faces)

        self.num_faces = len(faces)
        self.num_halfedges = 3 * self.num_faces

        self.tails_to_halfedges = faces.flatten()
        self.tips_to_halfedges = faces[:, tensor([2, 0, 1])].flatten()
        self.halfedges_to_faces = arange(self.num_halfedges).reshape(self.num_faces, 3)

    def angles_to_laplacian(self, alphas: Tensor):
        raise NotImplementedError
    
    def areas_to_mass_matrix(self, As: Tensor) -> Tensor:
        raise NotImplementedError

    def embedding_to_angles(self, fs: Tensor) -> Tensor:
        """Computes angles across from each halfedge

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * num_halfedges list of interior angles
        """
        return self.halfedge_vectors_to_angles(self.embedding_to_halfedge_vectors(fs))
    
    def embedding_to_com(self, fs: Tensor) -> Tensor:
        """Computes the center of mass of a uniform density surface

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * 3 list of centers of mass
        """
        batch_dims = fs.shape[:-2]
        fs_by_face = fs.permute(-2, *range(len(batch_dims)), -1)[self.faces].permute(*range(2, 2 + len(batch_dims)), 0, 1, -1)
        centroids = fs_by_face.mean(dim=-2)
        As = self.embedding_to_face_areas(fs)
        com = (As.unsqueeze(-1) * centroids / As.sum()).sum(dim=-2)
        return com
    
    def embedding_to_face_areas(self, fs: Tensor) -> Tensor:
        """Computes face areas

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * num_faces list of face areas
        """
        return norm(self.embedding_to_face_normals(fs, keep_scale=True), dim=-1) / 2
    
    def embedding_to_face_normals(self, fs: Tensor, keep_scale: bool = False) -> Tensor:
        """Computes outward pointing face normals, either unit length or scaled by twice the face area

        Returns:
            batch_dims * num_faces * 3 list of face normals
        """
        return self.halfedge_vectors_to_face_normals(self.embedding_to_halfedge_vectors(fs), keep_scale)
    
    def embedding_to_halfedge_vectors(self, fs: Tensor) -> Tensor:
        """Computes vectors pointing from halfedge tails to halfedge tips, where num_halfedges = 3 * num_faces

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * num_halfedges * 3 list of vertex differences
        """
        batch_dims = fs.shape[:-2]
        permuted_fs = fs.permute(-2, *range(len(batch_dims)), -1)
        es = (permuted_fs[self.tips_to_halfedges] - permuted_fs[self.tails_to_halfedges]).permute(*range(1, 1 + len(batch_dims)), 0, -1)
        return es
    
    def group_halfedge_vectors_by_face(self, es: Tensor) -> Tensor:
        """Organizes halfedge vectors, counterclockwise around each face

        Args:
            es (Tensor): batch_dims * num_halfedges * 3 list of halfedge vectors

        Returns:
            batch_dims * num_faces * 3 * 3 list of halfedge vectors
        """
        batch_dims = es.shape[:-2]
        es_by_face = es.permute(-2, *range(len(batch_dims)), -1)[self.halfedges_to_faces].permute(*range(1, 1 + len(batch_dims)), 0, -1)
        return es_by_face
    
    def group_halfedge_lengths_by_face(self, ls: Tensor) -> Tensor:
        """Organizes halfedge lengths, counterclockwise around each face

        Args:
            ls (Tensor): batch_dims * num_halfedges list of halfedge lengths

        Returns:
            batch_dims * num_faces * 3 list of halfedge lengths
        """
        batch_dims = ls.shape[:-1]
        ls_by_face = ls.permute(-1, *range(len(batch_dims)))[self.halfedges_to_faces].permute(*range(1, 1 + len(batch_dims)), 0)
        return ls_by_face
    
    def halfedge_vectors_to_angles(self, es: Tensor) -> Tensor:
        """Computes angles across from each halfedge

        Args:
            es (Tensor): batch_dims * num_halfedges * 3 list of halfedge vectors

        Returns:
            batch_dims * num_halfedges list of interior angles
        """
        es_by_face = self.group_halfedge_lengths_by_face(es)
        e_kjs = -es_by_face[..., tensor([1, 2, 0]), :]
        e_kis = es_by_face[..., tensor([2, 0, 1]), :]
        cos_alphas = (e_kjs * e_kis).sum(dim=-1) / (norm(e_kjs, dim=-1, keepdims=True) * norm(e_kis, dim=-1, keepdims=True))
        t = tensor(1., dtype=cos_alphas.dtype, device=cos_alphas.device)
        alphas = arccos(minimum(maximum(cos_alphas, -t), t))
        return alphas

    def halfedge_vectors_to_face_areas(self, es: Tensor) -> Tensor:
        """Computes face areas

        Args:
            es (Tensor): batch_dims * num_halfedges * 3 list of halfedge vectors
        
        Returns:
            batch_dims * num_faces list of face areas
        """
        return norm(self.halfedge_vectors_to_face_normals(es, keep_scale=True), dim=-1) / 2

    def halfedge_vectors_to_face_normals(self, es: Tensor, keep_scale: bool = False) -> Tensor:
        """Computes outward pointing face normals, either unit length or scaled by twice the face area

        Args:
            es (Tensor): batch_dims * num_halfedges * 3 list of halfedge vectors

        Returns:
            batch_dims * num_faces * 3 list of face normals
        """
        es_by_face = self.group_halfedge_vectors_by_face(es)
        Ns = cross(es_by_face[..., 0, :], -es_by_face[..., 2, :])

        if keep_scale:
            return Ns
        
        Ns = Ns / norm(Ns, dim=-1, keepdims=True)
        return Ns

    def halfedge_vectors_to_metric(self, es: Tensor) -> Tensor:
        """Computes discrete metric (halfedge lengths)

        Args:
            es (Tensor): batch_dims * num_halfedges * 3 list of halfedge vectors

        Returns:
            batch_dims * num_halfedges list of halfedge lengths
        """
        return norm(es, dim=-1)
    
    def metric_to_angles(self, ls: Tensor) -> Tensor:
        """Computes angles across from each halfedge using law of cosines

        Args:
            ls (Tensor): batch_dims * num_halfedges * 3 list of halfedge lengths

        Returns:
            batch_dims * num_halfedges list of interior angles
        """
        ls_by_face = self.group_halfedge_lengths_by_face(ls)
        l_ijs = ls_by_face
        l_jks = ls_by_face[..., tensor([1, 2, 0])]
        l_kis = ls_by_face[..., tensor([2, 0, 1])]
        cos_alphas = (l_jks ** 2 + l_kis ** 2 - l_ijs ** 2) / (2 * l_jks * l_kis)
        t = tensor(1., dtype=cos_alphas.dtype, device=cos_alphas.device)
        alphas = arccos(minimum(maximum(cos_alphas, -t), t))
        return alphas
    
    def metric_to_face_areas(self, ls: Tensor) -> Tensor:
        """Computes face areas using Heron's formula

        Args:
            ls (Tensor): batch_dims * num_halfedges * 3 list of halfedge lengths

        Returns:
            batch_dims * num_faces list of face areas
        """
        ls_by_face = self.group_halfedge_lengths_by_face(ls)
        l_ijs = ls_by_face
        l_jks = ls_by_face[..., tensor([1, 2, 0])]
        l_kis = ls_by_face[..., tensor([2, 0, 1])]
        sps = (l_ijs + l_jks + l_kis) / 2
        As = sqrt(sps * (sps - l_ijs) * (sps - l_jks) * (sps - l_kis))
        return As
    