from torch import arange, arccos, cat, diff, float64, maximum, minimum, ones, sort, sparse_coo_tensor, sqrt, stack, tan, Tensor, tensor, zeros
from torch.linalg import cross, norm
from torch.nn import Module


class Manifold(Module):
    """Stores topological data for a manifold triangle mesh. Computes objects for geometry processing when geometric data (vertex positions, edge lengths, etc.) are provided."""
    
    def __init__(self, faces: Tensor, dtype=float64):
        """
        Args:
            faces (Tensor): num_faces * 3 list of vertices per face
        """
        Module.__init__(self)
        self.register_buffer('faces', faces)

        self.num_vertices = faces.max().item() + 1
        self.num_faces = len(faces)
        self.num_halfedges = 3 * self.num_faces

        self.tails_to_halfedges = faces.flatten()
        self.tips_to_halfedges = faces[:, tensor([2, 0, 1])].flatten()
        self.halfedges_to_faces = arange(self.num_halfedges).reshape(self.num_faces, 3)
        self.faces_to_halfedges = arange(self.num_halfedges)

        self.halfedges_to_tails = sparse_coo_tensor(stack([self.tails_to_halfedges, arange(self.num_halfedges)]), ones(self.num_halfedges, dtype=float64))

        # Pair twin halfedges and find boundary halfedges (halfedges without a twin)
        tail_sorting = sort(self.tails_to_halfedges)
        neighborhood_start_indicators = diff(tail_sorting.values, prepend=tensor([-1]))
        neighborhood_start_idxs = arange(self.num_halfedges)[neighborhood_start_indicators == 1]
        neighborhood_degrees = diff(neighborhood_start_idxs, append=tensor([self.num_halfedges]))
        all_neighborhood_data = stack([tail_sorting.indices, self.tips_to_halfedges[tail_sorting.indices]], dim=-1)
        all_neighborhood_data = [all_neighborhood_data[start_idx:(start_idx + degree)] for start_idx, degree in zip(neighborhood_start_idxs, neighborhood_degrees)]
        
        self.halfedges_to_twins = -ones(self.num_halfedges, dtype=int)
        self.is_boundary_halfedge = zeros(self.num_halfedges, dtype=bool)
        for tail, neighborhood_data in enumerate(all_neighborhood_data):
            for halfedge_idx, tip in neighborhood_data:
                tip_neighborhood_data = all_neighborhood_data[tip]
                reflexive_data = tip_neighborhood_data[tip_neighborhood_data[:, 1] == tail]
                if len(reflexive_data) > 0:
                    twin_halfedge_idx = reflexive_data[0, 0]
                    self.halfedges_to_twins[halfedge_idx] = twin_halfedge_idx
                else:
                    self.is_boundary_halfedge[halfedge_idx] = True

        self.num_edges = (self.num_halfedges - self.is_boundary_halfedge.sum()) // 2
        self.euler_char = self.num_vertices - self.num_edges + self.num_faces

    def angles_to_laplacian(self, alphas: Tensor):
        """Computes cotan Laplacian from interior angles

        Args:
            alphas (Tensor): batch_dims * num_halfedges list of interior angles, across from each halfedge

        Returns:
            batch_dims * num_vertices * num_vertices list of Laplacians with same sparsity pattern
        """
        cot_alphas = 1 / tan(alphas)
        row_idxs = cat([self.tails_to_halfedges, self.tips_to_halfedges, self.tails_to_halfedges, self.tips_to_halfedges])
        col_idxs = cat([self.tips_to_halfedges, self.tails_to_halfedges, self.tails_to_halfedges, self.tips_to_halfedges])
        values = cat([cot_alphas / 2, cot_alphas / 2, -cot_alphas / 2, -cot_alphas / 2], dim=-1)
        L = sparse_coo_tensor(stack([row_idxs, col_idxs]), values.permute(-1, *range(len(values.shape[:-1])))).coalesce()
        return L
    
    def areas_to_mass_matrix(self, As: Tensor) -> Tensor:
        raise NotImplementedError

    def embedding_to_angles(self, fs: Tensor) -> sparse_coo_tensor:
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
        centroids = fs[..., self.faces, :].mean(dim=-2)
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
        return fs[..., self.tips_to_halfedges, :] - fs[..., self.tails_to_halfedges, :]
    
    def embedding_to_laplacian(self, fs: Tensor) -> sparse_coo_tensor:
        """Computes cotan Laplacian from vertex positions

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * num_vertices * num_vertices list of Laplacians with same sparsity pattern
        """
        return self.angles_to_laplacian(self.embedding_to_angles(fs))
    
    def embedding_to_metric(self, fs: Tensor) -> Tensor:
        """Computes discrete metric (halfedge lengths)

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * num_halfedges list of halfedge lengths
        """
        return self.halfedge_vectors_to_metric(self.embedding_to_halfedge_vectors(fs))
    
    def halfedge_vectors_to_angles(self, es: Tensor) -> Tensor:
        """Computes angles across from each halfedge

        Args:
            es (Tensor): batch_dims * num_halfedges * 3 list of halfedge vectors

        Returns:
            batch_dims * num_halfedges list of interior angles
        """
        es_by_face = es[..., self.halfedges_to_faces, :]
        e_kjs = -es_by_face[..., tensor([1, 2, 0]), :]
        e_kis = es_by_face[..., tensor([2, 0, 1]), :]
        cos_alphas = (e_kjs * e_kis).sum(dim=-1) / (norm(e_kjs, dim=-1) * norm(e_kis, dim=-1))
        t = tensor(1., dtype=cos_alphas.dtype, device=cos_alphas.device)
        alphas = arccos(minimum(maximum(cos_alphas, -t), t))
        alphas = alphas.flatten(start_dim=-2)[..., self.faces_to_halfedges]
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
        es_by_face = es[..., self.halfedges_to_faces, :]
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
        l_ijs = ls[..., self.halfedges_to_faces]
        l_jks = l_ijs[..., tensor([1, 2, 0])]
        l_kis = l_ijs[..., tensor([2, 0, 1])]
        cos_alphas = (l_jks ** 2 + l_kis ** 2 - l_ijs ** 2) / (2 * l_jks * l_kis)
        t = tensor(1., dtype=cos_alphas.dtype, device=cos_alphas.device)
        alphas = arccos(minimum(maximum(cos_alphas, -t), t))
        alphas = alphas.flatten(start_dim=-2)[..., self.faces_to_halfedges]
        return alphas
    
    def metric_to_face_areas(self, ls: Tensor) -> Tensor:
        """Computes face areas using Heron's formula

        Args:
            ls (Tensor): batch_dims * num_halfedges * 3 list of halfedge lengths

        Returns:
            batch_dims * num_faces list of face areas
        """
        ls_by_face = ls[..., self.halfedges_to_faces]
        l_ijs, l_jks, l_kis = ls_by_face.permute(-1, *range(len(ls_by_face.shape[:-1])))
        sps = (l_ijs + l_jks + l_kis) / 2
        As = sqrt(sps * (sps - l_ijs) * (sps - l_jks) * (sps - l_kis))
        return As
    