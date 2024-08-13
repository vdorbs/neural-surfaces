from torch import arange, arccos, cat, diff, eye, float64, maximum, minimum, ones, sort, sparse_coo_tensor, sqrt, stack, tan, Tensor, tensor, zeros
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
        self.tips_to_halfedges = faces[:, tensor([1, 2, 0])].flatten()
        self.halfedges_to_faces = arange(self.num_halfedges).reshape(self.num_faces, 3)
        self.faces_to_halfedges = arange(self.num_halfedges)

        self.halfedges_to_tails = sparse_coo_tensor(stack([self.tails_to_halfedges, arange(self.num_halfedges)]), ones(self.num_halfedges, dtype=dtype))

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
            num_vertices * num_vertices * batch_dims Laplacians, where first two dimensions are sparse
        """
        cot_alphas = 1 / tan(alphas)
        row_idxs = cat([self.tails_to_halfedges, self.tips_to_halfedges, self.tails_to_halfedges, self.tips_to_halfedges])
        col_idxs = cat([self.tips_to_halfedges, self.tails_to_halfedges, self.tails_to_halfedges, self.tips_to_halfedges])
        values = cat([cot_alphas / 2, cot_alphas / 2, -cot_alphas / 2, -cot_alphas / 2], dim=-1)
        L = sparse_coo_tensor(stack([row_idxs, col_idxs]), values.permute(-1, *range(len(values.shape[:-1])))).coalesce()
        return L
    
    def embedding_and_face_vectors_to_vertex_divs(self, fs: Tensor, vs: Tensor, use_diag_mass: bool = False) -> Tensor:
        """Computes vertexwise divergence of a tangent vector-valued function defined on faces
        
        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions
            vs (Tensor): batch_dims * num_faces * 3 list of tangent vectors per face

        Returns:
            batch_dims * num_vertices list of divergences per vertex
        """
        es = self.embedding_to_halfedge_vectors(fs)
        es_by_face = es[..., self.halfedges_to_faces, :]
        Ns = self.halfedge_vectors_to_face_normals(es, keep_scale=True)
        face_As = norm(Ns, dim=-1) / 2
        Ns = Ns / (2 * face_As).unsqueeze(-1)

        rot_es_by_face = cross(Ns.unsqueeze(-2), es_by_face)
        div_vs = -(self.halfedges_to_tails @ (rot_es_by_face * vs.unsqueeze(-2)).sum(dim=-1)[..., tensor([1, 2, 0])].flatten(start_dim=-2)[..., self.faces_to_halfedges]) / 2

        if use_diag_mass:
            vertex_As = self.face_areas_to_vertex_areas(face_As)
            div_vs = div_vs / vertex_As
        else:
            raise NotImplementedError
        
        return div_vs
    
    def embedding_and_vertex_signal_to_face_derivatives(self, fs: Tensor, phis: Tensor, vector_valued) -> Tensor:
        """Computes facewise gradient of a scalar or vector-valued function defined on vertices

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions
            phis (Tensor): batch_dims * num_vertices list of function values per vertex or batch_dims * num_vertices * d list of vectors per vertex
            vector_valued (bool): whether or not phis has a feature dimension d

        Returns:
            batch_dims * num_faces * 3 list of gradients per face or batch_dims * num_faces * d * 3 list of Jacobians per face
        """
        es = self.embedding_to_halfedge_vectors(fs)
        es_by_face = es[..., self.halfedges_to_faces, :]
        Ns = self.halfedge_vectors_to_face_normals(es, keep_scale=True)
        As = norm(Ns, dim=-1) / 2
        Ns = Ns / (2 * As).unsqueeze(-1)

        basis_grads_by_face = (cross(Ns.unsqueeze(-2), es_by_face) / (2 * As.unflatten(-1, (self.num_faces, 1, 1))))[..., tensor([1, 2, 0]), :]
        
        if vector_valued:
            phis_by_face = phis[..., self.tails_to_halfedges, :][..., self.halfedges_to_faces, :]
            jac_phis = (phis_by_face.unsqueeze(-1) * basis_grads_by_face.unsqueeze(-2)).sum(dim=-3)
            return jac_phis
        
        phis_by_face = phis[..., self.tails_to_halfedges][..., self.halfedges_to_faces]
        grad_phis = (phis_by_face.unsqueeze(-1) * basis_grads_by_face).sum(dim=-2)
        return grad_phis
    
    def embedding_and_vertex_values_to_face_grads(self, fs: Tensor, phis: Tensor) -> Tensor:
        """Computes facewise gradient of a function defined on vertices

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions
            phis (Tensor): batch_dims * num_vertices list of function values per vertex

        Returns:
            batch_dims * num_faces * 3 list of gradients per face
        """
        return self.embedding_and_vertex_signal_to_face_derivatives(fs, phis, vector_valued=False)
    
    def embedding_and_vertex_vectors_to_face_jacs(self, fs: Tensor, vs: Tensor) -> Tensor:
        """Computes facewise Jacobian of a tangent vector-valued function defined on vertices

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions
            phis (Tensor): batch_dims * num_vertices * d list of tangent vectors per vertex

        Returns:
            batch_dims * num_faces * d * 3 list of Jacobians per face
        """
        return self.embedding_and_vertex_signal_to_face_derivatives(fs, vs, vector_valued=True)

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
    
    def embedding_to_mass_matrix(self, fs: Tensor, use_diag_mass: bool = False) -> sparse_coo_tensor:
        """Computes mass matrix from vertex positions

        Args:
            fs (Tensor): batch_dims * num_vertices list of vertex positions
            use_diag_mass (bool): whether mass matrix is diagonal/lumped

        Returns:
            num_vertices * num_vertices * batch_dims list of mass matrices, where first two dimensions are sparse
        """
        return self.face_areas_to_mass_matrix(self.embedding_to_face_areas(fs), use_diag_mass=use_diag_mass)
    
    def embedding_to_metric(self, fs: Tensor) -> Tensor:
        """Computes discrete metric (halfedge lengths)

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * num_halfedges list of halfedge lengths
        """
        return self.halfedge_vectors_to_metric(self.embedding_to_halfedge_vectors(fs))
    
    def face_areas_to_mass_matrix(self, As: Tensor, use_diag_mass: bool = False) -> sparse_coo_tensor:
        """Computes mass matrix from face areas

        Args:
            As (Tensor): batch_dims * num_faces list of face areas
            use_diag_mass (bool): whether mass matrix is diagonal/lumped

        Returns:
            num_vertices * num_vertices * batch_dims list of mass matrices, where first two dimensions are sparse
        """
        if use_diag_mass:
            vertex_As = self.face_areas_to_vertex_areas(As)
            indices = arange(self.num_vertices)
            indices = stack([indices, indices])
            M = sparse_coo_tensor(indices, vertex_As, is_coalesced=True)
        else:
            template_values = ((eye(3, dtype=As.dtype) + ones(3, 3, dtype=As.dtype)) / 12).flatten()
            batch_dims = As.shape[:-1]
            values = (As.unsqueeze(-1) * template_values.reshape(tuple(1 for _ in batch_dims) + (1, 9))).flatten(start_dim=-2)
            row_idxs = self.faces.repeat_interleave(3, dim=-1).flatten()
            col_idxs = self.faces.repeat(1, 3).flatten()
            M = sparse_coo_tensor(stack([row_idxs, col_idxs]), values).coalesce()

        return M
        
    def face_areas_to_vertex_areas(self, As: Tensor) -> Tensor:
        """Distributes the area of each face to each of its vertices

        Args:
            As (Tensor): batch_dims * num_faces list of face areas

        Returns:
            batch_dims * num_vertices list of vertex areas
        """
        As_by_halfedges = (As.repeat_interleave(3, dim=-1) / 3)[..., self.faces_to_halfedges]
        vertex_As = self.halfedges_to_tails @ As_by_halfedges
        return vertex_As
    
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
    