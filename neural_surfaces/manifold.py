from __future__ import annotations
from mpmath import clsin
from neural_surfaces.utils import factorize, plane_to_sphere, sparse_solve
from torch import arange, arccos, arcsin, atan2, cat, clamp, cos, cumsum, diagonal, diff, exp, eye, float64, inf, log, maximum, minimum, multinomial, ones, pi, rand, rand_like, randn, sort, sparse_coo_tensor, sqrt, stack, tan, Tensor, tensor, zeros, zeros_like
from torch.linalg import det, cross, inv, norm, solve, svd
from torch.nn import Module
from torch.sparse import spdiags
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Union


class Manifold(Module):
    """Stores topological data for a manifold triangle mesh. Computes objects for geometry processing when geometric data (vertex positions, edge lengths, etc.) are provided."""
    
    def __init__(self, faces: Tensor, dtype=float64):
        """
        Args:
            faces (Tensor): num_faces * 3 list of vertices per face
        """
        Module.__init__(self)
        self.register_buffer('faces', faces.clone())

        self.num_vertices = faces.max().item() + 1
        self.num_faces = len(faces)
        self.num_halfedges = 3 * self.num_faces

        self.register_buffer('tails_to_halfedges', faces.clone().flatten())
        self.register_buffer('tips_to_halfedges', faces.clone()[:, tensor([1, 2, 0])].flatten())
        self.halfedges_to_faces = arange(self.num_halfedges).reshape(self.num_faces, 3)
        self.faces_to_halfedges = arange(self.num_halfedges)

        col_idxs = arange(self.num_halfedges, device=self.tails_to_halfedges.device)
        values = ones(self.num_halfedges, dtype=dtype, device=self.tails_to_halfedges.device)
        self.register_buffer('halfedges_to_tails', sparse_coo_tensor(stack([self.tails_to_halfedges, col_idxs]), values))

        # Pair twin halfedges and find boundary halfedges (halfedges without a twin)
        tail_sorting = sort(self.tails_to_halfedges)
        neighborhood_start_indicators = diff(tail_sorting.values, prepend=tensor([-1], device=self.tails_to_halfedges.device))
        neighborhood_start_idxs = arange(self.num_halfedges, device=neighborhood_start_indicators.device)[neighborhood_start_indicators == 1]
        neighborhood_degrees = diff(neighborhood_start_idxs, append=tensor([self.num_halfedges], device=neighborhood_start_idxs.device))
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

        self.num_edges = (self.num_halfedges + self.is_boundary_halfedge.sum()) // 2
        self.euler_char = (self.num_vertices - self.num_edges + self.num_faces).item()

        boundary_halfedges = stack([self.tails_to_halfedges[self.is_boundary_halfedge], self.tips_to_halfedges[self.is_boundary_halfedge]], dim=-1)
        boundary_vertices = set(boundary_halfedges[:, 0].tolist())
        boundary_loops = []
        while len(boundary_vertices) > 0:
            boundary_loop = [boundary_vertices.pop()]
            loop_incomplete = True
            while loop_incomplete:
                curr_vertex = boundary_loop[-1]
                next_vertex = boundary_halfedges[boundary_halfedges[:, 0] == curr_vertex][0, 1].item()
                if next_vertex == boundary_loop[0]:
                    loop_incomplete = False
                else:
                    boundary_vertices.remove(next_vertex)
                    boundary_loop.append(next_vertex)

            boundary_loops.append(tensor(boundary_loop))

        self.boundary_loops = boundary_loops

        is_interior_vertex = ones(self.num_vertices, dtype=bool)
        for boundary_loop in self.boundary_loops:
            is_interior_vertex[boundary_loop] = False
        self.is_interior_vertex = is_interior_vertex
        self.boundary_vertices = arange(self.num_vertices)[~self.is_interior_vertex]

    def angles_to_local_delaunay(self, alphas: Tensor) -> Tensor:
        """Computes whether or not halfedges satisfy local Delaunay condition

        Note:
            An interior halfedge satisfies local Delaunay condition if the pair of angles opposite it and its twin sum to pi or less

        Args:
            alphas (Tensor): batch_dims * num_halfedges list of interior angles, across from each halfedge

        Returns:
            batch_dims * num_halfedges list of booleans
        """
        angle_sums = alphas + alphas[..., self.halfedges_to_twins]
        local_delaunay = angle_sums <= pi
        local_delaunay[self.is_boundary_halfedge] = True
        return local_delaunay

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

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * num_faces * 3 list of face normals
        """
        return self.halfedge_vectors_to_face_normals(self.embedding_to_halfedge_vectors(fs), keep_scale)

    def embedding_to_frames(self, fs: Tensor) -> Tensor:
        """Computes a frame per face, with first two columns representing edges and third representing face normal
        
        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * num_faces * 3 * 3 list of frames per face
        """
        es = self.embedding_to_halfedge_vectors(fs)
        es_by_face = es[..., self.halfedges_to_faces, :]
        us = es_by_face[..., 0, :]
        vs = -es_by_face[..., 2, :]
        Ns = cross(us, vs)
        frames = stack([us, vs, Ns / sqrt(norm(Ns, dim=-1, keepdims=True))], dim=-1) # Partial normalization keeps frames well-conditioned
        return frames
    
    def embedding_to_halfedge_vectors(self, fs: Tensor) -> Tensor:
        """Computes vectors pointing from halfedge tails to halfedge tips, where num_halfedges = 3 * num_faces

        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions

        Returns:
            batch_dims * num_halfedges * 3 list of vertex differences
        """
        return fs[..., self.tips_to_halfedges, :] - fs[..., self.tails_to_halfedges, :]
    
    def embedding_to_heat_method_solver(self, fs: Tensor, use_diag_mass: bool = False, diff_coeff: float = 1.) -> Callable[[Tensor, bool, Tensor], Tensor]:
        """Precomputes data needed for heat method solver, which computes approximates distances

        Args:
            fs (Tensor): num_vertices * 3 list of vertex positions
            use_diag_mass (bool): whether mass matrix is diagonal/lumped
            diff_coeff (float): coefficient of diffusion, higher coefficient smooths distance more

        Returns:
            Function mapping a num_source list of vertex indices to a num_sources * num_vertices list of geodesic distances 
        """
        es = self.embedding_to_halfedge_vectors(fs)
        Ns = self.halfedge_vectors_to_face_normals(es, keep_scale=True)
        face_As = norm(Ns, dim=-1) / 2
        Ns = Ns / (2 * face_As).unsqueeze(-1)

        L = self.angles_to_laplacian(self.halfedge_vectors_to_angles(es))
        M = self.face_areas_to_mass_matrix(face_As, use_diag_mass=use_diag_mass)
        h = self.halfedge_vectors_to_metric(es).mean() ** 2
        A = (M - h * diff_coeff * L).coalesce()
        A_solver = factorize(A)

        L_def = self.laplacian_to_definite_laplacian(L)
        L_def_solver = factorize(-L_def)

        es_by_face = es[..., self.halfedges_to_faces, :]
        rot_es_by_face = cross(Ns.unsqueeze(-2), es_by_face)
        basis_grads_by_face = (rot_es_by_face / (2 * face_As.unflatten(-1, (self.num_faces, 1, 1))))[..., tensor([1, 2, 0]), :]

        if use_diag_mass:
            vertex_As = M.values()

        def heat_method_solver(source_idxs: Tensor, use_vertex_sources: bool = True, barys: Tensor = None) -> Tensor:
            """Computes geodesic distance from specified sources to all other vertices
            
            Args:
                source_idxs (Tensor): list of num_sources vertex/face indices
                use_vertex_sources (bool): whether to use vertices or points within faces as sources
                barys (Tensor): list of num_sources * 3 barycentric coordinates of sources if not using vertex sources

            Returns:
                num_sources * num_vertices list of geodesic distances
            """
            u_0s = zeros(self.num_vertices, len(source_idxs)).to(fs)
            for j, source_idx in enumerate(source_idxs):
                if use_vertex_sources:
                    u_0s[source_idx, j] = 1.
                else:
                    u_0s[self.faces[source_idx], j] = barys[j]

            u_hs = A_solver(M @ u_0s)
            u_hs_by_face = u_hs[self.tails_to_halfedges][self.halfedges_to_faces]
            eiko_field = (u_hs_by_face.unsqueeze(-1) * basis_grads_by_face.unsqueeze(-2)).sum(dim=-3)
            eiko_field = -eiko_field / norm(eiko_field, dim=-1, keepdims=True)
            eiko_field = eiko_field.transpose(0, 1)
            div_eiko_field = -(self.halfedges_to_tails @ (rot_es_by_face.unsqueeze(0) * eiko_field.unsqueeze(-2)).sum(dim=-1)[..., tensor([1, 2, 0])].flatten(start_dim=-2)[:, self.faces_to_halfedges].T) / 2
            
            free_dists = -L_def_solver(div_eiko_field[1:])
            dists = cat([zeros(1, len(source_idxs)).to(free_dists), free_dists])
            dists = dists - dists.min(dim=0, keepdims=True).values

            return dists

        return heat_method_solver
    
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
    
    def embedding_to_samples(self, fs: Tensor, num_samples: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes uniform samples from surface
        
        Args:
            fs (Tensor): batch_dims * num_vertices * 3 list of vertex positions
            num_samples (int): number of samples computed per batch element

        Returns:
            batch_dims * num_samples list of face indices, batch_dims * num_samples * 3 list of barycentric coordinates, and batch_dims * num_samples * 3 list of samples
        """
        batch_dims = fs.shape[:-2]
        face_As = self.embedding_to_face_areas(fs)
        flat_face_As = face_As.flatten(end_dim=len(batch_dims) - 1)
        flat_face_idxs = multinomial(flat_face_As, num_samples, replacement=True)
        face_idxs = flat_face_idxs.reshape(batch_dims + (num_samples,))

        bary_is = 1 - sqrt(1 - rand(batch_dims + (num_samples,), dtype=fs.dtype, device=fs.device))
        bary_js = (1 - bary_is) * rand_like(bary_is)
        bary_ks = 1 - bary_is - bary_js
        barys = stack([bary_is, bary_js, bary_ks], dim=-1)

        flat_barys = barys.flatten(end_dim=max(len(batch_dims) - 1, 0))
        flat_fs_by_faces = fs[..., self.faces, :].flatten(end_dim=max(len(batch_dims) - 1, 0))
        
        if face_idxs.shape == (num_samples,):
            flat_face_idxs = [flat_face_idxs]
            flat_barys = [flat_barys]
            flat_fs_by_faces = [flat_fs_by_faces]

        flat_samples = stack([(fs_by_faces_row[face_idxs_row] * barys_row.unsqueeze(-1)).sum(dim=-2) for fs_by_faces_row, face_idxs_row, barys_row in zip(flat_fs_by_faces, flat_face_idxs, flat_barys)])
        samples = flat_samples.reshape(batch_dims + (num_samples, 3))
        
        return face_idxs, barys, samples

    def embedding_to_sphere_embedding(self, fs: Tensor, flatten_iters: int, layout_iters: int, center_iters: int, verbose: bool = False, vertex_to_remove: int = 0, flatten_step_size: float = 0.5, use_diag_mass: bool = False, layout_eps: float = 1e-8, center_tol: float = 1e-12):
        As = self.embedding_to_face_areas(fs)
        disk = self.remove_vertex(vertex_to_remove)
        kept_fs = cat([fs[:vertex_to_remove], fs[(vertex_to_remove + 1):]])
        ls = disk.embedding_to_metric(kept_fs)
        flat_ls, _, _ = disk.metric_to_flat_metric(ls, flatten_iters, False, flatten_step_size, verbose)
        uvs = disk.metric_to_spectral_conformal_parametrization(flat_ls, layout_iters, use_diag_mass, verbose, layout_eps)
        sphere_fs = plane_to_sphere(uvs)
        sphere_fs = cat([tensor([[0, 0, 1]], dtype=float), sphere_fs])
        sphere_fs = -sphere_fs
        sphere_fs = self.sphere_embedding_and_face_areas_to_centered_sphere_embedding(sphere_fs, As, center_iters, center_tol, verbose)
        R = self.embeddings_to_rotation(sphere_fs, fs)
        sphere_fs = sphere_fs @ R.T
        return sphere_fs

    def embedding_to_tutte_parametrization(self, fs: Tensor) -> Tensor:
        """Computes parametrizations of disk topology surfaces using Tutte embedding

        Note:
            Boundary is mapped to the unit circle

        Args:
            fs (Tensor): num_vertices * 3 list of vertex positions

        Returns:
            num_vertices * 2 list of planar vertex positions
        """
        assert len(self.boundary_loops) == 1
        boundary_vertices = self.boundary_loops[0]
        boundary_fs = fs[boundary_vertices]
        boundary_ls = norm(diff(cat([boundary_fs, boundary_fs[..., :1, :]], dim=-2), dim=-2), dim=-1)
        boundary_ls = boundary_ls / boundary_ls.sum(dim=-1)
        boundary_zs = exp(2 * pi * cumsum(boundary_ls, dim=-1) * 1j)
        boundary_zs = cat([boundary_zs[..., -1:], boundary_zs[..., :-1]], dim=-1)
        boundary_uvs = stack([boundary_zs.real, boundary_zs.imag], dim=-1)

        L = self.embedding_to_laplacian(fs)
        L_int = self.laplacian_to_interior_laplacian(L)
        L_int_solver = factorize(-L_int)

        x_bc = zeros_like(fs[..., :2])
        x_bc[..., boundary_vertices, :] = boundary_uvs
        y = L @ x_bc
        x_int = -L_int_solver(-(L @ x_bc)[..., self.is_interior_vertex, :])

        uvs = zeros_like(x_bc)
        uvs[..., boundary_vertices, :] = boundary_uvs
        uvs[..., self.is_interior_vertex, :] = x_int
        return uvs

    def embedding_to_vertex_normals(self, fs: Tensor, keep_scale: bool = False) -> Tensor:
        fs_by_face = fs[..., self.faces, :]
        crosses_by_face = cross(fs_by_face, fs_by_face[..., tensor([1, 2, 0]), :])
        vertex_Ns = self.halfedges_to_tails @ crosses_by_face[..., tensor([1, 2, 0]), :].flatten(start_dim=-3, end_dim=-2)[..., self.faces_to_halfedges, :] / 6
        
        if keep_scale:
            return vertex_Ns

        vertex_Ns = vertex_Ns / norm(vertex_Ns, dim=-1, keepdims=True)
        return vertex_Ns

    def embeddings_to_rotation(self, fs: Tensor, target_fs: Tensor) -> Tensor:
        normalized_fs = fs - self.embedding_to_com(fs)
        normalized_fs = normalized_fs * sqrt(4 * pi / self.embedding_to_face_areas(normalized_fs).sum())

        normalized_target_fs = target_fs - self.embedding_to_com(target_fs)
        normalized_target_fs = normalized_target_fs * sqrt(4 * pi / self.embedding_to_face_areas(normalized_target_fs).sum())

        U, _, V_T = svd(normalized_target_fs.transpose(-2, -1) @ normalized_fs)
        R = U @ V_T
        return R
    
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
            indices = arange(self.num_vertices, device=As.device)
            indices = stack([indices, indices])
            M = sparse_coo_tensor(indices, vertex_As, size=(self.num_vertices, self.num_vertices), is_coalesced=True)
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
    
    def flip_halfedge(self, halfedge: int, ls: Optional[Tensor] = None, flip_type: str = 'euclidean') -> Optional[Tensor]:
        """Updates topological/geometric data to implement a non-boundary halfedge flip

        Note:
            If no discrete metric is provided, only topological data is updated

        Args:
            halfedge (int): index of halfedge to flip
            ls (Tensor): batch_dims * num_halfedges list of halfedge lengths
            flip_type (str): whether halfedge lengths are updated with an intrinsic Euclidean flip ('euclidean') or an intrinsic Ptolemy flip ('ptolemy')

        Returns:
            batch_dims * num_halfedges list of new discrete metric
        """
        tail = self.tails_to_halfedges[halfedge].item()
        tip = self.tips_to_halfedges[halfedge].item()
        
        face = (arange(self.num_faces)[(self.halfedges_to_faces == halfedge).any(dim=-1)])[0].item()
        selector = self.faces[face] == tail
        opp_vertex = (self.faces[face][tensor([2, 0, 1])][selector])[0].item()
        next_halfedge = (self.halfedges_to_faces[face][tensor([1, 2, 0])][selector])[0].item()
        next_next_halfedge = (self.halfedges_to_faces[face][tensor([2, 0, 1])][selector])[0].item()

        twin_halfedge = self.halfedges_to_twins[halfedge].item()
        twin_face = (arange(self.num_faces)[(self.halfedges_to_faces == twin_halfedge).any(dim=-1)])[0].item()
        twin_selector = self.faces[twin_face] == tail
        twin_opp_vertex = (self.faces[twin_face][tensor([1, 2, 0])][twin_selector])[0].item()
        twin_next_halfedge = (self.halfedges_to_faces[twin_face][twin_selector])[0].item()
        twin_next_next_halfedge = (self.halfedges_to_faces[twin_face][tensor([1, 2, 0])][twin_selector])[0].item()

        if ls is not None:
            if flip_type == 'euclidean':
                angle = 0
                for a, b, c in [[next_next_halfedge, halfedge, next_halfedge], [twin_halfedge, twin_next_halfedge, twin_next_next_halfedge]]:
                    cos_half_angle = (ls[a] ** 2 + ls[b] ** 2 - ls[c] ** 2) / (2 * ls[a] * ls[b])
                    cos_half_angle = clamp(cos_half_angle, -1, 1)
                    angle = angle + arccos(cos_half_angle)

                a = ls[next_next_halfedge]
                b = ls[twin_next_halfedge]
                c_squared = a ** 2 + b ** 2 - 2 * a * b * cos(angle)
                c_squared = clamp(c_squared, min=0)
                new_l = sqrt(c_squared)

            elif flip_type == 'ptolemy':
                new_l = (ls[next_halfedge] * ls[twin_next_halfedge] + ls[next_next_halfedge] * ls[twin_next_next_halfedge]) / ls[halfedge]

        self.tails_to_halfedges[halfedge] = twin_opp_vertex
        self.tails_to_halfedges[twin_halfedge] = opp_vertex
        self.halfedges_to_tails = sparse_coo_tensor(stack([self.tails_to_halfedges, arange(self.num_halfedges)]), ones(self.num_halfedges, dtype=self.halfedges_to_tails.dtype))

        self.tips_to_halfedges[halfedge] = opp_vertex
        self.tips_to_halfedges[twin_halfedge] = twin_opp_vertex

        self.faces[face] = tensor([tail, twin_opp_vertex, opp_vertex])
        self.faces[twin_face] = tensor([tip, opp_vertex, twin_opp_vertex])

        self.halfedges_to_faces[face] = tensor([twin_next_halfedge, halfedge, next_next_halfedge])
        self.halfedges_to_faces[twin_face] = tensor([next_halfedge, twin_halfedge, twin_next_next_halfedge])

        self.faces_to_halfedges[self.halfedges_to_faces[face]] = arange(3 * face, 3 * (face + 1))
        self.faces_to_halfedges[self.halfedges_to_faces[twin_face]] = arange(3 * twin_face, 3 * (twin_face + 1))

        if ls is not None:
            ls = ls.clone()

            new_ls = tensor([ls[twin_next_halfedge], new_l, ls[next_next_halfedge]])
            new_twin_ls = tensor([ls[next_halfedge], new_l, ls[twin_next_next_halfedge]])

            ls[self.halfedges_to_faces[face]] = new_ls
            ls[self.halfedges_to_faces[twin_face]] = new_twin_ls

            return ls
            
    def frames_to_singular_values(self, base_frames: Tensor, deform_frames: Tensor) -> Tensor:
        """Computes singular values for each face through a deformation, excluding the singular value associated with normal vectors
        
        Args:
            base_frames (Tensor): batch_size * num_faces * 3 * 3 list of frames for base surface
            deform_frames (Tensor): batch_size * num_faces * 3 * 3 list of frames for deformed surface

        Returns:
            batch_size * num_faces * 2 list of singular values in descending order per face
        """
        jacs = deform_frames @ inv(base_frames)
        jac_prods = jacs @ jacs.transpose(-2, -1)
        traces = diagonal(jac_prods, dim1=-2, dim2=-1).sum(dim=-1)
        dets = det(jac_prods)

        base_normal_mags = norm(base_frames[..., -1], dim=-1)
        deform_normal_mags = norm(deform_frames[..., -1], dim=-1)
        ratios = (deform_normal_mags / base_normal_mags) ** 2

        bs = ratios - traces
        cs = dets / ratios
        discrims = bs ** 2 - 4 * cs
        discrims = maximum(discrims, tensor(0).to(discrims))
        squared_sigma_highs = (-bs + sqrt(discrims)) / 2
        squared_sigma_lows = (-bs - sqrt(discrims)) / 2
        squared_sigmas = stack([squared_sigma_highs, squared_sigma_lows], dim=-1)
        squared_sigmas = maximum(squared_sigmas, tensor(0).to(squared_sigmas))
        sigmas = sqrt(squared_sigmas)
        return sigmas

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

    def laplacian_to_conformal_energy_operator(self, L: sparse_coo_tensor) -> sparse_coo_tensor:
        """Computes sparse matrix characterizing conformal energy quadratic form from Spectral Conformal Parametrization
        
        Args:
            L (sparse_coo_tensor): num_vertices * num_vertices sparse Laplacian matrix

        Returns:
            (2 * num_vertices) * (2 * num_vertices) real representation of conformal energy operator as a sparse matrix
        """
        offsets = tensor([[0, 1], [0, 1]])
        indices = L.indices()
        indices = 2 * indices.repeat_interleave(2, dim=-1) + offsets.repeat(1, indices.shape[-1])
        values = L.values().repeat_interleave(2, dim=-1)
        L_comp = sparse_coo_tensor(indices, values, (2 * self.num_vertices, 2 * self.num_vertices), is_coalesced=True)
        
        offsets = tensor([0, 1, 1, 0, 0, 1, 1, 0])
        template_values = tensor([1, -1, -1, 1], dtype=values.dtype) / 4
        all_indices = []
        all_values = []
        for boundary_loop in self.boundary_loops:
            cycled_boundary_loop = cat([boundary_loop[1:], boundary_loop[:1]])
            indices = stack([boundary_loop, cycled_boundary_loop, cycled_boundary_loop, boundary_loop], dim=-1)
            indices = indices.reshape(-1, 2).repeat(1, 2).reshape(-1, 8) # each row has form i, j, i, j, j, i, j, i
            indices = (2 * indices + offsets).reshape(-1, 2).T
            values = template_values.repeat(len(boundary_loop))

            all_indices.append(indices)
            all_values.append(values)

        indices = cat(all_indices, dim=-1)
        values = cat(all_values)
        A_comp = sparse_coo_tensor(indices, values, (2 * self.num_vertices, 2 * self.num_vertices)).coalesce()

        L_conf = -L_comp / 2 - A_comp
        return L_conf

    def laplacian_to_definite_laplacian(self, L: sparse_coo_tensor, idx: int = 0) -> sparse_coo_tensor:
        """Removes specified row and column of Laplacian matrix, eliminating the zero eigenvalue

        Args:
            L (sparse_coo_tensor): num_vertices * num_vertices spase Laplacian matrix
            idx (int): index of row and column to be removed

        Returns:
            (num_vertices - 1) * (num_vertices - 1) block of sparse Laplacian matrix
        """
        indices = L.indices()
        is_free = (indices != idx).all(dim=0)
        free_indices = indices[:, is_free]
        free_indices -= (free_indices > idx).to(int)
        
        free_values = L.values()[is_free]
        L_def = sparse_coo_tensor(free_indices, free_values, (self.num_vertices - 1, self.num_vertices - 1), is_coalesced=True)
        return L_def

    def laplacian_to_interior_laplacian(self, L: sparse_coo_tensor) -> sparse_coo_tensor:
        """Computes block of Laplacian matrix corresponding only to interior vertices

        Args:
            L (sparse_coo_tensor): num_vertices * num_vertices sparse Laplacian matrix

        Returns:
            num_interior_vertices * num_interior_vertices block of sparse Laplacian matrix
        """
        indices = L.indices()
        is_int = (indices.unsqueeze(-1) != self.boundary_vertices.reshape(1, 1, -1)).all(dim=-1).all(dim=0)
        int_indices = indices[:, is_int]
        int_indices = int_indices - (int_indices.unsqueeze(-1) > self.boundary_vertices.reshape(1, 1, -1)).sum(dim=-1)

        int_values = L.values()[is_int]
        num_int_vertices = self.num_vertices - len(self.boundary_vertices)
        L_int = sparse_coo_tensor(int_indices, int_values, (num_int_vertices, num_int_vertices), is_coalesced=True)
        return L_int
    
    def metric_to_angles(self, ls: Tensor) -> Tensor:
        """Computes angles across from each halfedge using law of cosines

        Args:
            ls (Tensor): batch_dims * num_halfedges list of halfedge lengths

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

    def metric_to_delaunay_metric(self, ls: Tensor, flip_type: str = 'euclidean') -> Tensor:
        if flip_type == 'euclidean':
            is_local_delaunay = self.angles_to_local_delaunay(self.metric_to_angles(ls))
        elif flip_type == 'ptolemy':
            is_local_delaunay = self.metric_to_local_ideal_delaunay(ls)

        flips = []
        while not is_local_delaunay.all():
            idx = arange(len(is_local_delaunay))[~is_local_delaunay][0].item()
            flips.append(idx)
            ls = self.flip_halfedge(idx, ls, flip_type)

            if flip_type == 'euclidean':
                is_local_delaunay = self.angles_to_local_delaunay(self.metric_to_angles(ls))
            elif flip_type == 'ptolemy':
                is_local_delaunay = self.metric_to_local_ideal_delaunay(ls)

        return ls, flips
    
    def metric_to_face_areas(self, ls: Tensor) -> Tensor:
        """Computes face areas using Heron's formula

        Args:
            ls (Tensor): batch_dims * num_halfedges list of halfedge lengths

        Returns:
            batch_dims * num_faces list of face areas
        """
        ls_by_face = ls[..., self.halfedges_to_faces]
        l_ijs, l_jks, l_kis = ls_by_face.permute(-1, *range(len(ls_by_face.shape[:-1])))
        sps = (l_ijs + l_jks + l_kis) / 2
        As = sqrt(sps * (sps - l_ijs) * (sps - l_jks) * (sps - l_kis))
        return As

    def metric_to_flat_metric(self, ls: Tensor, num_iters: int, flip_halfedges: bool = False, step_size: float = 0.5, verbose: bool = False) -> Union[Tuple[Tensor, Tensor, float], Tuple[Tensor, Tensor, float, List[int]]]:
        """Optimizes log conformal factors to flatten a discrete metric, as in Conformal Equivalence of Triangle Meshes (CETM) or Discrete Conformal Equivalence of Polyhedral Surfaces (CEPS)

        Note:
            If flip_halfedges is True, Ptolemy flips are performed during optimization as in CEPS
            If flip_halfedges is False, this method does not implement convex extension of energy as in CETM. If log conformal factors violate triangle inequality during optimization, construction of the Laplacian will fail.

        Args:
            ls (Tensor): num_halfedges list of halfedge lengths
            num_iters (int): number of flattening iterations
            flip_halfedges (bool): whether or not halfedges are flipped with Ptolemy flips as in CEPS
            step_size (float): step size of Newton's method
            verbose (bool): whether or not to print optimization information

        Returns:
            num_halfedges list of new halfedge lengths, num_vertices list of log conformal factors generating new lengths, maximum interior angle defect from 2 * pi, and optionally, list of flips performed
        """
        assert len(self.boundary_loops) > 0
        ls = ls.clone()
        us = zeros(self.num_vertices).to(ls)

        iterator = range(num_iters)
        if verbose:
            iterator = tqdm(iterator)

        if flip_halfedges:
            flips = []

        for _ in iterator:
            if flip_halfedges:
                ls, new_flips =  self.metric_to_delaunay_metric(ls, flip_type='ptolemy')
                flips += new_flips
                
            lambdas = 2 * log(ls)
            angles = self.metric_to_angles(ls)
            angles_by_face = angles[self.halfedges_to_faces]
            angle_sums = self.halfedges_to_tails @ angles_by_face[:, tensor([1, 2, 0])].flatten()[self.faces_to_halfedges]

            grad = (2 * pi - angle_sums[self.is_interior_vertex]) / 2
            hess = -self.angles_to_laplacian(angles) / 2
            hess_int = self.laplacian_to_interior_laplacian(hess)

            if verbose:
                # lobachevsky_angles = tensor([float(clsin(2, 2 * angle)) / 2 for angle in angles.tolist()])
                # energy = (angles * lambdas).sum() / 2 + lobachevsky_angles.sum()# -pi * us[self.faces].sum() / 2 + pi * us[self.is_interior_vertex].sum()
                # iterator.set_description(f'energy={energy.item()} max_int_angle_defect={grad.abs().max().item()}')
                iterator.set_description(f'max_int_angle_defect={grad.abs().max().item()}')

            d_int = sparse_solve(hess_int, grad.unsqueeze(-1))[:, 0]
            diff_us = zeros_like(us)
            diff_us[self.is_interior_vertex] = -step_size * d_int
            lambdas = lambdas + diff_us[self.tips_to_halfedges] + diff_us[self.tails_to_halfedges]
            ls = exp(lambdas / 2)

            us = us - diff_us

        if flip_halfedges:
            return ls, us, grad.abs().max().item(), flips

        return ls, us, grad.abs().max().item()

    def metric_to_local_ideal_delaunay(self, ls: Tensor) -> Tensor:
        """Computes whether or not interior halfedges satisfy local ideal Delaunay condition from Discrete Conformal Equivalence of Polyhedral Surfaces

        Note:
            Boundary halfedges are automatically assigned True

        Args:
            ls (Tensor): batch_dims * num_halfedges list of halfedge lengths

        Returns:
            batch_dims * num_halfedges list of booleans
        """
        l_ijs = ls
        
        ls_by_face = ls[..., self.halfedges_to_faces]
        l_jks = ls_by_face[..., tensor([1, 2, 0])].flatten(start_dim=-2)[..., self.faces_to_halfedges]
        l_kis = ls_by_face[..., tensor([2, 0, 1])].flatten(start_dim=-2)[..., self.faces_to_halfedges]

        l_ils = l_jks[..., self.halfedges_to_twins]
        l_ljs = l_kis[..., self.halfedges_to_twins]

        local_ideal_delaunay = (l_ijs ** 2) * (l_jks * l_kis + l_ils * l_ljs) < (l_ils * l_kis + l_jks * l_ljs) * (l_ils * l_jks + l_kis * l_ljs)
        local_ideal_delaunay[self.is_boundary_halfedge] = True
        return local_ideal_delaunay

    def metric_to_spectral_conformal_parametrization(self, ls: Tensor, num_iters: int, use_diag_mass: bool = False, verbose: bool = False, eps: float = 1e-8) -> Tensor:
        L = self.angles_to_laplacian(self.metric_to_angles(ls))
        L_conf = self.laplacian_to_conformal_energy_operator(L)
        I = spdiags(ones(2 * self.num_vertices, dtype=ls.dtype), tensor(0), shape=L_conf.shape)    
        L_conf_eps = (L_conf + eps * I).coalesce()
        L_conf_eps_solver = factorize(L_conf_eps)
    
        M = (1 + 0j) * self.face_areas_to_mass_matrix(self.metric_to_face_areas(ls), use_diag_mass=use_diag_mass)

        us, vs = randn(2, self.num_vertices).to(ls)
        x = us + 1j * vs

        iterator = range(num_iters)
        if verbose:
            iterator = tqdm(iterator)

        for _ in iterator:
            y = M @ x

            # Complex to real representation
            y_comp = stack([y.real, -y.imag, y.imag, y.real], dim=-1).reshape(2 * self.num_vertices, 2)
            next_x_comp = L_conf_eps_solver(y_comp)
            # Real representation to complex
            next_x = next_x_comp[::2, 0] - 1j * next_x_comp[::2, 1]

            next_x = next_x - (M @ next_x).sum() / M.sum()
            next_x = next_x / sqrt((next_x.conj() * (M @ next_x)).sum().abs())
            x = next_x

            if verbose:
                x_comp = stack([x.real, -x.imag, x.imag, x.real], dim=-1).reshape(2 * self.num_vertices, 2)
                lhs_comp = L_conf_eps @ x_comp
                lhs = lhs_comp[::2, 0] - 1j * lhs_comp[::2, 1]

                eigval_comp = x_comp.T @ lhs_comp
                eigval = eigval_comp[0, 0]
                
                rhs = eigval * (M @ x)
                
                residual = norm(lhs - rhs)
                iterator.set_description(f'residual={residual.item()}')

        uvs = stack([x.real, x.imag], dim=-1)
        return uvs

    def remove_vertex(self, idx: int = 0) -> Manifold:
        """Removes a single vertex along with all incident faces
        
        Args:
            idx (int): index of vertex to be removed

        Returns:
            Manifold with face removed
        """
        faces = self.faces.clone()
        faces = faces[(faces != idx).all(dim=-1)]
        faces -= (faces > idx).to(int)
        return Manifold(faces)

    def sphere_embedding_and_embedding_to_plot_data(self, sphere_fs: Tensor, fs: Tensor, y_up: bool = False) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor], bool]:
        """Prepares data needed to render a mesh textured with a checkerboard pattern
        
        Args:
            sphere_fs (Tensor): num_vertices * 3 list of vertex positions on the unit sphere
            fs (Tensor): num_vertices * 3 list of vertex positions to plot
            y_up (bool): whether x points right, y points up, z points forward or x points forward, y points right, z points up
        """
        Ns = self.embedding_to_vertex_normals(fs)
        
        if y_up:
            us = (atan2(sphere_fs[:, 0], sphere_fs[:, 2]) + pi) / (2 * pi)
            vs = (arcsin(sphere_fs[:, 1]) + pi / 2) / pi
        else:
            us = (atan2(sphere_fs[:, 1], sphere_fs[:, 0]) + pi) / (2 * pi)
            vs = (arcsin(sphere_fs[:, 2]) + pi / 2) / pi
        uvs = stack([us, vs], dim=-1)

        return fs.cpu(), self.faces.cpu(), Ns.cpu(), uvs.cpu(), True, None, y_up

    def sphere_embedding_and_face_areas_to_centered_sphere_embedding(self, sphere_fs: Tensor, As: Tensor, num_iters: int, tol: float = 1e-12, verbose: bool = False) -> Tensor:
        batch_dims = sphere_fs.shape[:-2]
        I = eye(3).to(sphere_fs).reshape(tuple(1 for _ in batch_dims) + (3, 3))
        ps = (As / As.sum()).unsqueeze(-1)
        
        iterator = range(num_iters)
        if verbose:
            iterator = tqdm(iterator)

        for _ in iterator:
            centroids = sphere_fs[self.faces].mean(dim=-2)
            centroids = centroids / norm(centroids, dim=-1, keepdims=True)
            com = (ps * centroids).sum(dim=-2)
            com_norm = norm(com, dim=-1)
            if (com_norm < tol).all():
                break

            jac = 2 * (ps.unsqueeze(-1) * (I - centroids.unsqueeze(-1) * centroids.unsqueeze(-2))).sum(dim=-3)
            inv_center = -solve(jac, com)
            
            while norm(inv_center, dim=-1).max() >= 1:
                inv_center = inv_center / 2

            new_com_norm = inf
            num_divs = 0
            while (new_com_norm > com_norm).any():
                new_sphere_fs = sphere_fs + inv_center.unsqueeze(-2)
                new_sphere_fs = new_sphere_fs / (norm(new_sphere_fs, dim=-1, keepdims=True) ** 2)
                new_sphere_fs = (1 - norm(inv_center, dim=-1) ** 2).reshape(batch_dims + (1, 1)) * new_sphere_fs
                new_sphere_fs = new_sphere_fs + inv_center.unsqueeze(-2)
                new_centroids = new_sphere_fs[self.faces].mean(dim=-2)
                new_centroids /= norm(new_centroids, dim=-1, keepdims=True)
                new_com = (ps * new_centroids).sum(dim=-2)
                new_com_norm = norm(new_com, dim=-1, keepdims=True)
                inv_center = inv_center / 2

                num_divs += 1

            num_divs -= 1 

            if verbose:
                iterator.set_description(f'{num_divs} {new_com_norm.max().item()}')

            sphere_fs = new_sphere_fs

        return sphere_fs

    def sphere_embedding_to_locator(self, sphere_fs: Tensor) -> Callable[[Tensor], Tuple[Tensor, Tensor]]:
        """Precomputes data needed for sphere locator, which computes face indices and barycentric coordinates for a spherical partition
        
        Args:
            sphere_fs: num_vertices * 3 list of vertex positions on the unit sphere

        Returns:
            Function mapping a batch_dim * 3 list of query vertex positions to a batch_dims list of face indices and a batch_dims * 3 list of barycentric coordinates
        """
        Ns = self.embedding_to_face_normals(sphere_fs)
        sphere_fs_by_face = sphere_fs[self.faces]
        halfspace_normals_by_face = cross(sphere_fs_by_face, sphere_fs_by_face[:, tensor([1, 2, 0]), :], dim=-1)
        halfspace_normals_by_face = halfspace_normals_by_face / norm(halfspace_normals_by_face, dim=-1, keepdims=True)

        def locator(query: Tensor) -> Tuple[Tensor, Tensor]:
            """Locates query points by face index and barycentric coordinates
            
            Args:
                query (Tensor): batch_dims * 3 list of query points on the sphere

            Returns:
                batch_dims list of face indices and batch_dims * 3 list of corresponding barycentric coordinates
            """
            batch_dims = query.shape[:-1]
            reshaped_halfspace_normals_by_face = halfspace_normals_by_face.reshape(tuple(1 for _ in batch_dims) + halfspace_normals_by_face.shape)
            reshaped_query = query.reshape(batch_dims + (1, 1, 3))
            query_halfspace_magnitudes_by_face = (reshaped_halfspace_normals_by_face * reshaped_query).sum(dim=-1)
            query_memberships_by_face = (query_halfspace_magnitudes_by_face >= 0).all(dim=-1)
            face_idxs = (arange(self.num_faces, device=query_memberships_by_face.device) * query_memberships_by_face).sum(dim=-1)

            query_Ns = Ns[face_idxs]
            query_sphere_fs_by_face = sphere_fs_by_face[face_idxs]
            query_proj_mag = (query_Ns * query_sphere_fs_by_face[..., 0, :]).sum(dim=-1, keepdims=True) / (query_Ns * query).sum(dim=-1, keepdims=True)
            proj_query = query_proj_mag * query

            query_subareas = norm(cross(query_sphere_fs_by_face - query_sphere_fs_by_face[..., tensor([1, 2, 0]), :], proj_query.unsqueeze(-2) - query_sphere_fs_by_face), dim=-1) / 2
            query_subareas = query_subareas[..., tensor([1, 2, 0])]
            barys = query_subareas / query_subareas.sum(dim=-1, keepdims=True)

            return face_idxs, barys
                
        return locator
    