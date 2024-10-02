from cholespy import CholeskySolverD, MatrixType
from http.server import SimpleHTTPRequestHandler
from io import BytesIO
from os import system
from socket import AF_INET, SOCK_DGRAM, socket
from socketserver import TCPServer
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized, spsolve
import torch
from torch import arange, arccos, chunk, clamp, cos, diff, float64, nan, pi, sin, sinc, sparse_coo_tensor, stack, Tensor, tensor, zeros_like
from torch.linalg import norm
from torchsparsegradutils import sparse_generic_solve
from trimesh.exchange.obj import load_obj
from typing import Callable, Dict, List, Tuple
from urllib.request import urlopen


armadillo = dict(name='armadillo', low='armadillo')
bunny = dict(name='bunny', low='bunny', high='bunny_hr')
cat = dict(name='cat', low='cat-low-resolution', high='cat')
fish = dict(name='fish', low='fish_low_resolution', high='fish')
goat_head = dict(name='goathead', high='goathead')
koala = dict(name='koala', low='koala_low_resolution', high='koala')
nefertiti = dict(name='nefertiti', low='nefertiti-lowres', high='nefertiti')
penguin = dict(name='penguin', low='penguin', high='penguin_hr')
plane = dict(name='plane', high='plane')
scorpion = dict(name='scorpion', low='scorpion_low_resolution', high='scorpion')
spot = dict(name='spot', low='spot_low_resolution', high='spot')
mesh_dicts = [armadillo, bunny, cat, fish, goat_head, koala, nefertiti, penguin, plane, scorpion, spot]
url_prefix = 'https://raw.githubusercontent.com/odedstein/meshes/master/objects'

class OdedSteinMeshes:
    """Class for accessing meshes from https://github.com/odedstein/meshes
    
    Available meshes are armadillo, bunny, cat, fish, goat_head, koala, nefertiti, penguin, plane, scorpion, and spot. Each mesh has a single component, no boundary, and Euler characteristic 2 (genus 0).

    .. code-block:: py
        meshes = OdedSteinMeshes()
        fs, faces = meshes.spot()
        // fs is num_vertices * 3 list of vertex positions
        // faces is num_faces * 3 list of vertices per face

        // Load all meshes
        data = []
        for name in meshes.names:
            fs, faces = getattr(meshes, name)()
            data.append((fs, faces))
    """
    def __init__(self, is_high_res: bool = False, preload: bool = False):
        """Creates class for accessing meshes from https://github.com/odedstein/meshes

        Args:
            is_high_res (bool): whether or not to load high resolution versions of meshes
            preload (bool): whether or not to load all meshes at construction
        """
        res = 'high' if is_high_res else 'low'

        self.names = []
        for mesh_dict in mesh_dicts:
            if res in mesh_dict:
                name = mesh_dict['name']
                self.names.append(name)
                url = f'{url_prefix}/{name}/{mesh_dict[res]}.obj'

                if preload:
                    fs, faces = load_obj_from_url(url)
                    setattr(self, name, lambda data=(fs, faces): data)
                else:
                    setattr(self, name, lambda url=url: load_obj_from_url(url))


def load_obj_from_url(url: str) -> Tuple[Tensor, Tensor]:
    """Loads mesh data from obj file at url

    Returns:
        num_vertices * 3 list of vertex positions and num_faces * 3 list of vertices per face
    """
    with urlopen(url) as response:
        buffer = BytesIO(response.read())
    mesh = load_obj(buffer, maintain_order=True)
    return tensor(mesh['vertices']), tensor(mesh['faces'])

def meshes_to_html(all_fs: List[List[Tensor]], all_faces: List[List[Tensor]], all_Ns: List[List[Tensor]], all_uvs: List[List[Tensor]], y_up: bool = True, mode: str = 'none') -> str:
    """Creates HTML string for rendering textured meshes with Babylon.js

    Note:
        num_vertices and num_faces can be different for each mesh
        If 'turbo' is selected as mode, v coordinate is unused
        If 'none' is selected as mode, both u and v coordinates are unused

    Args:
        all_fs (List[List[Tensor]]): num_rows list of num_cols lists of num_vertices * 3 lists of vertex positions
        all_faces (List[List[Tensor]]): num_rows list of num_cols lists of num_faces * 3 lists of vertices per face
        all_Ns (List[List[Tensor]]): num_rows list of num_cols list of num_vertices * 3 lists of vertex normals
        all_uvs (List[List[Tensor]]): num_rows list of num_cols lists of num_vertices * 2 lists of uv coordinates per vertex
        y_up (bool): whether x points right, y points up, z points forward or x points forward, y points right, z points up
        mode (str): whether rendered texture is 'checkerboard', 'turbo' (rainbow colormap), or 'none' (single color)

    Returns:
        HTML string, can be saved to a file or logged to a HTML-supported logger
    """

    num_rows = len(all_fs)
    num_cols = len(all_fs[0])

    perm = tensor([0, 1, 2]) if y_up else tensor([1, 2, 0])

    all_positions = [[fs[:, perm][faces].flatten().tolist() for fs,faces in zip(*rows)] for rows in zip(all_fs, all_faces)]
    all_indices = [[arange(3 * len(faces)).tolist() for faces in faces_row] for faces_row in all_faces]
    all_normals = [[Ns[:, perm][faces].flatten().tolist() for Ns, faces in zip(*rows)] for rows in zip(all_Ns, all_faces)]

    all_wrapped_uvs = []
    for rows in zip(all_uvs, all_faces):
        wrapped_uvs_row = []
        for uvs, faces in zip(*rows):
            us, vs = uvs.T
            
            us_by_face = us[faces]
            crosses_seam = (diff(us_by_face, dim=-1).abs() > 0.75).any(dim=-1)
            seam_crossing_us_by_face = us_by_face[crosses_seam]
            seam_crossing_us_by_face += (seam_crossing_us_by_face < 0.5)
            us_by_face[crosses_seam] = seam_crossing_us_by_face
            us_by_face /= 2

            vs_by_face = vs[faces]
            wrapped_uvs = stack([us_by_face.flatten(), vs_by_face.flatten()], dim=-1).flatten().tolist()
            wrapped_uvs_row.append(wrapped_uvs)

        all_wrapped_uvs.append(wrapped_uvs_row)

    all_uvs = all_wrapped_uvs

    with open('renderMeshes.js') as f:
        js_str = f.read()

    row_str = '<div class="row">' + ''.join(['<canvas></canvas>'] * num_cols) + '</div>'
    dom_str = ''.join([row_str] * num_rows)
    html_str = f"""<!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.babylonjs.com/babylon.js"></script>
                    <style>
                        canvas {{
                            width: {100 // num_cols}vw;
                            height: {100 // num_rows}vh;
                        }}

                        canvas#engineCanvas {{
                            width: 0;
                            height: 0;
                        }}

                        div.row {{
                            display: flex;
                        }}
                    </style>
                </head>
                <body>
                    <canvas id="engineCanvas"></canvas>
                    {dom_str}
                    <script>
                        {js_str}
                        renderMeshes({all_positions}, {all_indices}, {all_normals}, {all_uvs}, "{mode}");
                    </script>
                </body>
                </html>
                """
    
    return html_str

def mesh_trajectories_to_html(fs_traj: Tensor, faces: Tensor, Ns_traj: Tensor, uvs: Tensor, y_up: bool = True, mode: str = 'none', loop_mode: str = 'cycle') -> str:
    """Creates HTML string for rendering textured mesh animations with Babylon.js

    Note:
        If 'turbo' is selected as mode, v coordinate is unused
        If 'none' is selected as mode, both u and v coordinates are unused

    Args:
        fs_traj (Tensor): num_frames * num_vertices * 3 list of list of vertex positions per animation frame
        faces (Tensor): num_faces * 3 list of vertices per face
        Ns_traj (Tensor): num_frames * num_vertices * 3 list of list of vertex normals per animation frame
        uvs (Tensor): num_vertices * 2 list of uv coordinates per vertex
        y_up (bool): whether x points right, y points up, z points forward or x points forward, y points right, z points up
        mode (str): whether rendered texture is 'checkerboard', 'turbo' (rainbow colormap), or 'none' (single color)
        loop_mode (str): whether animation loops through a cycle with 'cycle' or reverses to the start with 'yoyo'

    Returns:
        HTML string, can be saved to a file or logged to a HTML-supported logger
    """

    perm = tensor([0, 1, 2]) if y_up else tensor([1, 2, 0])

    position_frames = fs_traj[..., perm].flatten(start_dim=-2).tolist()
    indices = faces.flatten().tolist()
    normal_frames = Ns_traj[..., perm].flatten(start_dim=-2).tolist()
    uvs = uvs.flatten().tolist()
    
    with open('renderMeshAnimation.js') as f:
        js_str = f.read()

    html_str = f"""<!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.babylonjs.com/babylon.js"></script>
                    <style>
                        canvas#renderCanvas {{
                            width: 100vw;
                            height: 100vh;
                        }}
                    </style>
                </head>
                <body>
                    <canvas id="renderCanvas"></canvas>
                    <script>
                        {js_str}
                        renderMeshAnimation({position_frames}, {indices}, {normal_frames}, {uvs}, "{mode}", "{loop_mode}");
                    </script>
                </body>
                </html>
                """
    
    return html_str

def point_cloud_trajectories_and_mesh_to_html(x_trajs: Tensor, radii: float, frame_rate: float, fs: Tensor, faces: Tensor, Ns: Tensor, y_up: bool = True) -> str:
    """Creates HTML string for rendering point cloud animations with Babylon.js
    
    Args:
        x_trajs (Tensor): num_frames * num_points * 3 list of list of point cloud positions per point cloud animation frame
        radii (float): radius of each sphere in point cloud
        frame_rate (float): frames per second for point cloud animation
        fs (Tensor): num_vertices * 3 list of vertex positions
        faces (Tensor): num_faces * 3 list of vertices per face
        Ns (Tensor): num_vertices * 3 list of vertex normals
        y_up (bool): whether x points right, y points up, z points forward or x points forward, y points right, z points up

    Returns:
        HTML string, can be saved to a file or logged to a HTML-supported logger
    """
    perm = tensor([0, 1, 2]) if y_up else tensor([1, 2, 0])
    
    frames = x_trajs[..., perm].tolist()
    positions = fs[:, perm].flatten().tolist()
    indices = faces.flatten().tolist()
    normals = Ns[:, perm].flatten().tolist()

    with open('renderPointCloudAnimation.js') as f:
        js_str = f.read()

    html_str = f"""<!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.babylonjs.com/babylon.js"></script>
                    <style>
                        canvas#renderCanvas {{
                            width: 100vw;
                            height: 100vh;
                        }}
                    </style>
                </head>
                <body>
                    <canvas id="renderCanvas"></canvas>
                    <script>
                        {js_str}
                        renderPointCloudAnimation({frames}, {radii}, {frame_rate}, {positions}, {indices}, {normals});
                    </script>
                </body>
                </html>
                """
    
    return html_str

def serve_html(html_str: str, serve_locally: bool = True, port: int = 8000):
    """Starts a server which serves a specified HTML string
    
    Args:
        html_str (str): HTML string to serve
        serve_locally (bool): whether server is visible from localhost (True) or local network IP address (False)
        port (int): server port
    """

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_str.encode())

    if serve_locally:
        ip = 'localhost'
    else:
        with socket(AF_INET, SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]

    with TCPServer((ip, port), Handler, bind_and_activate=False) as httpd:
        # Allow quicker startups after shutdowns
        httpd.allow_reuse_address = True
        httpd.server_bind()
        httpd.server_activate()

        print(f'Serving at http://{ip}:{port}')
        httpd.serve_forever()

def create_rectangular_mesh(num_rows: int, num_cols: int, is_2d: bool = False) -> Tuple[Tensor, Tensor]:
    """Creates a triangle mesh of a rectangle in either 2D or 3D
    
    Args:
        num_rows (int): number of vertices in vertical direction
        num_cols (int): number of vertices is horizontal direction
        is_2d (bool): whether or not vertices are 2D or 3D (with zero third coordinate)

    Returns:
        (num_rows * num_cols) * d list of vertex positions and ((num_rows - 1) * 2 * (num_cols - 1)) * 3 list of faces per vertex
    """
    xs = arange(num_cols, dtype=float64).repeat(num_rows)
    ys = arange(num_rows).repeat_interleave(num_cols)
    if is_2d:
        vertices = stack([xs, ys], dim=-1)
    else:
        zs = zeros_like(ys)
        vertices = stack([xs, ys, zs], dim=-1)

    faces = []
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            faces += [[num_rows * i + j, num_rows * i + j + 1, num_rows * i + num_cols + j]]
            faces += [[num_rows * i + j + 1, num_rows * i + num_cols + j + 1, num_rows * i + num_cols + j]]
    faces = tensor(faces)

    return vertices, faces

def factorize(A: sparse_coo_tensor) -> Callable[[Tensor], Tensor]:
    """Performs sparse Cholesky factorization to solve linear system AX = B

    Note:
        For multi-GPU systems, the matrices A and B must have device rank 0. To use a device of a different rank, set the CUDA_VISIBLE_DEVICES environment variable.
        If m > 128, the right hand side will be split into chunks, and each chunk will be processed separately.
    
    Args:
        A (sparse_coo_tensor): sparse n * n symmetric positive definite matrix

    Returns:
        Function mapping a dense n * m right-hand side matrix to a dense n * m solution matrix
    """
    chol_solver = CholeskySolverD(len(A), *A.indices(), A.values(), MatrixType.COO)

    def backbone_solver(A: sparse_coo_tensor, B: Tensor) -> Tensor:
        num_rhs = B.shape[-1]
        
        if num_rhs > 128:
            B_batches = chunk(B, (num_rhs // 128) + 1, dim=-1)
            X_batches = []
            for B_batch in B_batches:
                B_batch = B_batch.contiguous()
                X_batch = zeros_like(B_batch)
                chol_solver.solve(B_batch, X_batch)
                X_batches.append(X_batch)
            X = torch.cat(X_batches, dim=-1)

        else:
            X = zeros_like(B)
            chol_solver.solve(B, X)
        
        return X

    def solver(B: Tensor) -> Tensor:
        """Solves the linear system AX = B

        Args:
            B (Tensor): dense n * m right-hand side matrix

        Returns:
            Dense n * m solution matrix
        """
        return sparse_generic_solve(A, B, solve=backbone_solver, transpose_solve=backbone_solver)

    return solver

def sparse_solve(A: sparse_coo_tensor, B: Tensor) -> Tensor:
    """Solves linear system AX = B

    Note:
        Each function call performs Cholesky factorization of A. To solve the linear system many times with different right-hand sides, use factorize
        For multi-GPU systems, the matrices A and B must have device rank 0. To use a device of a different rank, set the CUDA_VISIBLE_DEVICES environment variable.
        If B has more than 128 columns, the right hand side will be split into chunks, and each chunk will be processed separately.
    
    Args:
        A (sparse_coo_tensor): sparse symmetric positive definite n * n matrix
        B (Tensor): dense n * m right-hand side matrix

    Returns:
        Dense n * m solution matrix
    """
    solver = factorize(A)
    return solver(B)

def ceps_parametrize(ceps_path, filename, output_filename) -> Tuple[Tensor, Tensor, Tensor]:
    """Runs spherical_uniformize from Discrete Conformal Equivalence of Polyhedral Surfaces (CEPS)

    Note:
        The file saved as output_filename stores the spherical parametrization as vertex texture quadruples, and the mesh may contain convex quadrilaterals.

    Args:
        ceps_path (str): path to CEPS directory
        filename (str): path to input .obj file
        output_filename (str): path for CEPS output

    Returns:
        num_vertices * 3 list of sphere vertex positions, num_vertices * 3 list of mesh vertex positions, and num_faces * 3 list of vertices per face
    """
    system(f'{ceps_path}/build/bin/spherical_uniformize {filename} --outputMeshFilename {output_filename}')
    with open(output_filename) as f:
        lines = f.read().split('\n')

    fs = []
    sphere_fs = []
    faces = []
    for line in lines:

        if line[:2] == 'v ':
            v = tensor(list(map(float, line.split(' ')[1:])))
            fs.append(v)

        elif line[:2] == 'vt':
            v = tensor(list(map(float, line.split(' ')[1:])))
            v = v[:3] / norm(v[:3])
            sphere_fs.append(v)

        elif line[:2] == 'f ':
            f = line.split(' ')[1:]
            f = [int(s.split('/')[0]) for s in f]
            if len(f) == 3:
                faces.append(tensor(f))
            elif len(f) == 4:
                i, j, k, l = f
                faces.append(tensor([i, j, l]))
                faces.append(tensor([k, l, j]))

    return stack(sphere_fs), stack(fs), stack(faces) - 1

def sphere_exp(x: Tensor, v: Tensor) -> Tensor:
    """Computes sphere exponential

    Args:
        x (Tensor): batch_dims * 3 list of base points on sphere
        v (Tensor): batch_dims * 3 list of tangent vectors to sphere at base points

    Returns:
        batch_dims * 3 list of translated points on sphere
    """
    norm_v = norm(v, dim=-1, keepdims=True)
    return cos(norm_v) * x + sinc(norm_v / pi) * v

def sphere_log(x: Tensor, y: Tensor, keep_scale: bool = True) -> Tensor:
    """Computes inverse of sphere exponential

    Args:
        x (Tensor) batch_dims * 3 list of base points on sphere
        y (Tensor) batch_dims * 3 list of non-antipodal points on sphere
        keep_scale (bool): whether logs have true scale (geodesic distance between x and y) or are normalized

    Returns:
        batch_dims * 3 list of tangent vectors to sphere at base points
    """
    cos_theta = (x * y).sum(dim=-1)
    cos_theta = clamp(cos_theta, -1, 1)
    theta = arccos(cos_theta)

    unit_logs = zeros_like(x)
    unit_logs[theta == pi] = nan
    idxs = (theta > 0) * (theta < pi)
    unit_logs[idxs] = (y[idxs] - cos_theta[idxs].unsqueeze(-1) * x[idxs]) / sin(theta[idxs]).unsqueeze(-1)

    if keep_scale:
        logs = theta.unsqueeze(-1) * unit_logs
        return logs

    return unit_logs  

def plane_to_sphere(z: Tensor) -> Tensor:
    """Stereographically project plane onto the unit sphere through the north pole
    
    Args:
        z (Tensor): batch_dims * 2 list of planar points

    Returns:
        batch_dims * 3 list of sphere points
    """
    squared_norm_z = norm(z, dim=-1, keepdims=True) ** 2
    p = torch.cat([2 * z, squared_norm_z - 1], dim=-1) / (squared_norm_z + 1)
    return p

def sphere_to_plane(p: Tensor) -> Tensor:
    """Stereographically project unit sphere onto the plane through the north pole
    
    Args:
        z (Tensor): batch_dims * 3 list of sphere points

    Returns:
        batch_dims * 2 list of planar points
    """
    z = p[..., :2] / (1 - p[..., -1:])
    return z
