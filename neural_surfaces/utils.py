from cholespy import CholeskySolverD, MatrixType
from cvxpy import Minimize, multiply, Problem, sum, sum_squares, Variable
from http.server import SimpleHTTPRequestHandler
from io import BytesIO
from os import system
from potpourri3d import EdgeFlipGeodesicSolver, face_areas, read_mesh
from pygeodesic.geodesic import PyGeodesicAlgorithmExact
from socket import AF_INET, SOCK_DGRAM, socket
from socketserver import TCPServer
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized, spsolve
import torch
from torch import arange, arccos, atan2, chunk, clamp, cos, cumsum, diag, diff, exp, float64, linspace, nan, ones, pi, searchsorted, sin, sinc, sparse_coo_tensor, sqrt, stack, Tensor, tensor, zeros_like
from torch.autograd import Function
from torch.linalg import cross, norm, solve
from torch.nn import Module
from torchsparsegradutils import sparse_generic_solve
from tqdm import tqdm
from trimesh.exchange.obj import load_obj
from typing import Callable, Dict, List, Optional, Tuple
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
    mesh = load_obj(buffer, maintain_order=True)['geometry']['geometry']
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
            faces += [[num_cols * i + j, num_cols * i + j + 1, num_cols * (i + 1) + j]]
            faces += [[num_cols * i + j + 1, num_cols * (i + 1) + j + 1, num_cols * (i + 1) + j]]
    faces = tensor(faces)

    return vertices, faces

def create_toroidal_mesh(num_rows: int, num_cols: int, inner_radius: float, outer_radius: float) -> Tuple[Tensor, Tensor, Tensor]:
    xs = arange(num_cols - 1, dtype=float64).repeat(num_rows - 1)
    ys = arange(num_rows - 1).repeat_interleave(num_cols - 1)
    torus_vertices = stack([xs, ys], dim=-1)
    torus_vertices /= tensor([[num_cols - 1, num_rows - 1]], dtype=float)

    thetas, phis = 2 * pi * torus_vertices.T
    inners = inner_radius * stack([cos(thetas), sin(thetas), zeros_like(thetas)], dim=-1)
    vertices = inners + outer_radius * stack([cos(phis) * cos(thetas), cos(phis) * sin(thetas), sin(phis)], dim=-1)

    faces = []
    for i in range(num_rows - 2):
        for j in range(num_cols - 2):
            faces += [[(num_cols - 1) * i + j, (num_cols - 1) * i + j + 1, (num_cols - 1) * (i + 1) + j]]
            faces += [[(num_cols - 1) * i + j + 1, (num_cols - 1) * (i + 1) + j + 1, (num_cols - 1) * (i + 1) + j]]

        j = num_cols - 2
        faces += [[(num_cols - 1) * i + j, (num_cols - 1) * i, (num_cols - 1) * (i + 1) + j]]
        faces += [[(num_cols - 1) * i, (num_cols - 1) * (i + 1), (num_cols - 1) * (i + 1) + j]]

    i = num_rows - 2
    for j in range(num_cols - 2):
        faces += [[(num_cols - 1) * i + j, (num_cols - 1)* i + j + 1, j]]
        faces += [[(num_cols - 1) * i + j + 1, j + 1, j]]

    i = num_rows - 2
    j = num_cols - 2
    faces += [[(num_cols - 1) * i + j, (num_cols - 1) * i, j]]
    faces += [[(num_cols - 1) * i, 0, j]]
    faces = tensor(faces)

    return torus_vertices, vertices, faces

def rectangle_locator(fs: Tensor, faces: Tensor, xs: Tensor) -> Tuple[Tensor, Tensor]:
    batch_dims = xs.shape[:-1]
    fs_by_face = fs[faces]
    es_by_face = diff(torch.cat([fs_by_face, fs_by_face[:, :1, :]], dim=-2), dim=-2)
    As = (es_by_face[:, 0, 0] * (-es_by_face[:, -1, 1]) - es_by_face[:, 0, 1] * (-es_by_face[:, -1, 0])).abs()

    reshaped_fs_by_face = fs_by_face.reshape(tuple(1 for _ in batch_dims) + fs_by_face.shape)
    reshaped_es_by_face = es_by_face.reshape(reshaped_fs_by_face.shape)
    reshaped_As = As.reshape(tuple(1 for _ in batch_dims) + As.shape)
    reshaped_xs = xs.reshape(batch_dims + (1, 1, 2))
    diff_xs = reshaped_xs - reshaped_fs_by_face

    subareas = (reshaped_es_by_face[..., 0] * diff_xs[..., 1] - diff_xs[..., 0] * reshaped_es_by_face[..., 1]).abs()
    barys = subareas / reshaped_As.unsqueeze(-1)
    barys = barys[..., tensor([1, 2, 0])]
    bary_sums = barys.sum(dim=-1)
    is_valid = (bary_sums - 1).abs() < 1e-12

    face_idxs = (arange(len(faces), device=is_valid.device) * is_valid).sum(dim=-1)
    barys = (barys * is_valid.unsqueeze(-1)).sum(dim=-2)
    return face_idxs, barys

def ceps_parametrize(ceps_path, filename, output_filename, use_original_triangulation: bool = False, timeout: int = 300) -> Tuple[Tensor, Tensor, Tensor]:
    """Runs spherical_uniformize from Discrete Conformal Equivalence of Polyhedral Surfaces (CEPS)

    Note:
        The file saved as output_filename stores the spherical parametrization as vertex texture quadruples, and the mesh may contain convex quadrilaterals.

    Args:
        ceps_path (str): path to CEPS directory
        filename (str): path to input .obj file
        output_filename (str): path for CEPS output
        use_original_triangulation (bool): whether or not to return data only from vertices that match input mesh

    Returns:
        num_vertices * 3 list of sphere vertex positions, num_vertices * 3 list of mesh vertex positions, and num_faces * 3 list of vertices per face
    """
    system(f'timeout {timeout} {ceps_path}/build/bin/spherical_uniformize {filename} --outputMeshFilename {output_filename}')
    with open(output_filename) as f:
        lines = f.read().split('\n')

    fs = []
    sphere_fs = []
    faces = []
    for line in lines:

        if line[:2] == 'v ':
            v = tensor(list(map(float, line.split(' ')[1:])), dtype=float64)
            fs.append(v)

        elif line[:2] == 'vt':
            v = tensor(list(map(float, line.split(' ')[1:])), dtype=float64)
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

    ceps_sphere_fs = stack(sphere_fs)
    ceps_fs = stack(fs)
    ceps_faces = stack(faces) - 1

    if use_original_triangulation:
        ceps_As = tensor(face_areas(ceps_fs, ceps_faces))
        ceps_com = (ceps_As.unsqueeze(-1) * ceps_fs[ceps_faces].mean(dim=-2) / ceps_As.sum()).sum(dim=-2)

        fs, faces = map(tensor, read_mesh(filename))
        As = tensor(face_areas(fs, faces))
        com = (As.unsqueeze(-1) * fs[faces].mean(dim=-2) / As.sum()).sum(dim=-2)
    
        ceps_fs += com - ceps_com
        idxs = norm(fs.unsqueeze(1) - ceps_fs.unsqueeze(0), dim=-1).min(dim=-1).indices
        sphere_fs = ceps_sphere_fs[idxs]

        return sphere_fs, fs, faces

    return ceps_sphere_fs, ceps_fs, ceps_faces

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

def bezier_c1(points: Tensor, tangents: Tensor) -> Callable[[Tensor], Tensor]:
    """Precompute data for piecewise cubic spline with once-differentiable transitions

    Args:
        points (Tensor): num_points * dim list of positions of knot points (transitions between pieces)
        tangents (Tensor): num_points * dim list of vectors tangent to curve at knot points

    Returns:
        Function that maps the interval [0, 1] to a curve
    """
    num_segments = len(points) - 1

    next_control_points = points[:-1] + tangents[:-1] / 3
    prev_control_points = points[1:] - tangents[1:] / 3
    control_points = stack([points[:-1], next_control_points, prev_control_points, points[1:]])

    starts = arange(num_segments, dtype=float, device=points.device).unsqueeze(-1)
    ends = starts + 1
    starts[0, 0] = -1e-12
    ends[-1, 0] = num_segments + 1e-12

    def f(ts: Tensor) -> Tensor:
        unit_ts = (num_segments * ts).unsqueeze(0)
        offset_ts = unit_ts - starts
        is_in_range = (starts < unit_ts) * (ends >= unit_ts)

        offset_ts = offset_ts.unsqueeze(-1)
        is_in_range = is_in_range.unsqueeze(-1)
        a, b, c, d = control_points.unsqueeze(-2)
        out = ((1 - offset_ts) ** 3) * a + 3 * ((1 - offset_ts) ** 2) * offset_ts * b + 3 * (1 - offset_ts) * (offset_ts ** 2) * c + (offset_ts ** 3) * d
        out = (is_in_range * out).sum(dim=0)
        return out

    return f

def bezier_c2(points: Tensor, init_tangent: Tensor, final_tangent: Tensor) -> Tuple[Callable[[Tensor], Tensor], Tensor]:
    """Precompute data for piecewise cubic spline with twice-differentiable transitions

    Note:
        Compared to piecewise cubic splines with once-differentiable transitions, this method only requires tangents at first and last knot points

    Args:
        points (Tensor): num_points * dim list of positions of knot points (transitions between pieces)
        init_tangent (Tensor): tangent vector of size dim at first knot point
        final_tangent (Tensor): tangent vector of size dim at last knot point

    Returns:
        Function that maps the interval [0, 1] to a curve and num_points * dim tangent vectors at each knot point
    """
    num_points = len(points)
    d = diag(4 * ones(num_points - 2, dtype=float, device=points.device) / 3)
    sup_d = diag(ones(num_points - 3, dtype=float, device=points.device) / 3, diagonal=1)
    A = sup_d + d + sup_d.T

    b = points[2:] - points[:-2]
    b[0] = b[0] - init_tangent / 3
    b[-1] = b[-1] - final_tangent / 3

    tangents = solve(A, b)
    tangents = torch.cat([init_tangent.unsqueeze(0), tangents, final_tangent.unsqueeze(0)])
    
    return bezier_c1(points, tangents), tangents

def bezier_c2_periodic(points: Tensor) -> Tuple[Callable[[Tensor], Tensor], Tensor]:
    """Precompute data for periodic piecewise cubic spline with twice-differentiable transitions

    Note:
        Compared to piecewise cubic splines with once- or twice-differentiable transitions, this method requires no tangents

    Args:
        points (Tensor): num_points * dim positions of knot points (transitions between pieces), with identical first and last knot points

    Returns:
        Function that maps the interval [0, 1] to a curve and num_points * dim tangent vectors at each knot point
    """
    num_points = len(points) - 1
    d = diag(4 * ones(num_points, dtype=float, device=points.device) / 3)
    sup_d = diag(ones(num_points - 1, dtype=float, device=points.device) / 3, diagonal=1)
    A = sup_d + d + sup_d.T
    A[0, -1] = 1 / 3
    A[-1, 0] = 1 / 3

    b = points[1:] - torch.cat([points[-2:], points[1:-2]])

    tangents = solve(A, b)
    tangents = torch.cat([tangents, tangents[:1]])
    
    return bezier_c1(points, tangents), tangents

def bezier_c2_normal(points: Tensor, normals: Tensor) -> Tuple[Callable[[Tensor], Tensor], Tensor]:
    """Precompute data for piecewise cubic spline with twice-differentiable transitions, with normal-constrained initial and final tangents, and intermediate tangents optimized for normality

    Note:
        Compared to piecewise cubic splines with once- or twice-differentiable transitions, this method requires no tangents

    Args:
        points (Tensor): num_points * dim list of positions of knot points (transitions between pieces)
        normals (Tensor): num_points * dim list of vectors to which tangents at knot points should be normal

    Returns:
        Function that maps the interval [0, 1] to a curve and num_points * dim tangent vectors at each knot point
    """
    device = points.device
    points = points.cpu()
    normals = normals.cpu()

    num_points = len(points)
    d = diag(4 * ones(num_points, dtype=float) / 3)
    sup_d = diag(ones(num_points - 1, dtype=float) / 3, diagonal=1)
    A = sup_d + d + sup_d.T
    A = A[1:-1]

    b = points[2:] - points[:-2]

    tangents = Variable(points.shape)
    obj = sum_squares(sum(multiply(normals, tangents), axis=1))
    cons = [A @ tangents == b, normals[0] @ tangents[0] == 0, normals[-1] @ tangents[-1] == 0]
    prob = Problem(Minimize(obj), cons)
    prob.solve()

    tangents = tensor(tangents.value).to(device)
    return bezier_c1(points.to(device), tangents), tangents

def resample_curve(dense_ts, dense_ys, N):
    """Samples curve at inputs to space out outputs approximately uniformly

    Note:
        num_points should be greater than N
        The first and last entries of the new inputs match the first and last entries of dense_ts

    Args:
        dense_ts (Tensor): num_points list of inputs in ascending order
        dense_ys (Tensor): num_points * dim list of corresponding outputs
        N (int): number of new inputs to return

    Returns:
        list of N inputs
    """
    unnorm_pdf = norm(diff(dense_ys, dim=0), dim=-1) / diff(dense_ts)
    unnorm_cdf = cumsum(unnorm_pdf, dim=0)
    cdf = zeros_like(dense_ts)
    cdf[1:] = unnorm_cdf / unnorm_cdf[-1]

    cdf_eps = cdf.clone()
    cdf_eps[0] = -1e-12
    cdf_eps[-1] = 1 + 1e-12

    quantiles = linspace(0, 1, N).to(dense_ts)
    idxs = searchsorted(cdf_eps, quantiles)
    left_cdfs = cdf[idxs - 1]
    right_cdfs = cdf[idxs]
    interps = (quantiles - left_cdfs) / (right_cdfs - left_cdfs)

    left_ts = dense_ts[idxs - 1]
    right_ts = dense_ts[idxs]
    new_ts = (1 - interps) * left_ts + interps * right_ts
    
    return new_ts

from torch.linalg import cross

def exact_geodesic_pairwise_distances(fs, faces, face_idxs, barys, verbose: bool = False):
    curr_faces = faces.clone()
    curr_fs = fs.clone()
    remaining_face_idxs = face_idxs.clone()
    remaining_barys = barys.clone()

    while len(remaining_face_idxs) > 0:
        dividing_face_idx = remaining_face_idxs[0]
        dividing_barys = remaining_barys[0]
        dividing_vertex = (dividing_barys.unsqueeze(-1) * curr_fs[curr_faces[dividing_face_idx]]).sum(dim=-2)

        remaining_face_idxs = remaining_face_idxs[1:]
        remaining_barys = remaining_barys[1:]

        select_fs = curr_fs[curr_faces[dividing_face_idx]]
        is_matching = remaining_face_idxs == dividing_face_idx
        matching_fs = (remaining_barys[is_matching].unsqueeze(-1) * select_fs.unsqueeze(0)).sum(dim=-2)

        curr_fs = torch.cat([curr_fs, dividing_vertex.unsqueeze(0)])
        i, j, k = curr_faces[dividing_face_idx]
        l = len(curr_fs) - 1
        new_faces = tensor([[i, j, l], [j, k, l], [k, i, l]])
        select_fs = curr_fs[new_faces]
        select_edges = diff(torch.cat([select_fs, select_fs[:, :1, :]], dim=-2), dim=-2)
        matching_edges = matching_fs.reshape(-1, 1, 1, 3) - select_fs.unsqueeze(0)
        matching_subareas = norm(cross(select_edges.unsqueeze(0), matching_edges, dim=-1), dim=-1) / 2
        areas = norm(cross(select_edges[:, 0, :], -select_edges[:, 2, :], dim=-1), dim=-1) / 2
        matching_barys = matching_subareas / areas.unsqueeze(-1)
        matching_barys = matching_barys[:, :, tensor([1, 2, 0])]
        valid_bary_idxs = (matching_barys.sum(dim=-1) - 1).abs().min(dim=-1).indices
        matching_barys = matching_barys[arange(len(matching_barys)), valid_bary_idxs]

        curr_faces = torch.cat([curr_faces[:dividing_face_idx], curr_faces[(dividing_face_idx + 1):], new_faces])
        remaining_face_idxs -= (remaining_face_idxs > dividing_face_idx).to(int)
        remaining_face_idxs[is_matching] = len(curr_faces) - 3 + valid_bary_idxs
        remaining_barys[is_matching] = matching_barys

    # solver = EdgeFlipGeodesicSolver(curr_fs.numpy(), curr_faces.numpy())
    # new_idxs = arange(len(fs), len(curr_fs))

    # if verbose:
    #     iterator = tqdm(new_idxs)
    # else:
    #     iterator = new_idxs

    # pairwise_dists = []
    # for i in iterator:
    #     dists = []
    #     for j in new_idxs:
    #         if i >= j:
    #             dists.append(tensor(0, dtype=float))
    #         else:
    #             path = tensor(solver.find_geodesic_path(i, j))
    #             dists.append(norm(diff(path, dim=0), dim=-1).sum())

    #     pairwise_dists.append(stack(dists))
    
    # pairwise_dists = stack(pairwise_dists)
    # pairwise_dists += pairwise_dists.T
    
    solver = PyGeodesicAlgorithmExact(curr_fs.numpy(), curr_faces.numpy())
    new_idxs = arange(len(fs), len(curr_fs))

    pairwise_dists = []
    for idx in tqdm(new_idxs):
        dists, _ = solver.geodesicDistances(idx.unsqueeze(0).numpy(), None)
        dists = tensor(dists)[new_idxs]
        pairwise_dists.append(dists)

    pairwise_dists = stack(pairwise_dists)
    return curr_fs, curr_faces, pairwise_dists

class CholeskySolver:
    def __init__(self, A: sparse_coo_tensor):
        self.chol_solver = CholeskySolverD(len(A), *A.indices(), A.values(), MatrixType.COO)

    def solve(self, B: Tensor) -> Tensor:
        B = B.clone()
        num_rhs = B.shape[-1]

        if num_rhs > 128:
            B_batches = chunk(B, (num_rhs // 128) + 1, dim=-1)
            X_batches = []
            for B_batch in B_batches:
                B_batch = B_batch.contiguous()
                X_batch = zeros_like(B_batch)
                self.chol_solver.solve(B_batch, X_batch)
                X_batches.append(X_batch)
            X = torch.cat(X_batches, dim=-1)

        else:
            B = B.contiguous()
            X = zeros_like(B)
            self.chol_solver.solve(B, X)

        return X

class FactorizedSolve(Function):
    @staticmethod
    def forward(ctx, B: Tensor, chol_solver: CholeskySolver):
        ctx.chol_solver = chol_solver
        X = chol_solver.solve(B)
        return X

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        chol_solver = ctx.chol_solver
        grad_inputs = chol_solver.solve(grad_outputs)
        return grad_inputs, None

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
    chol_solver = CholeskySolver(A)

    def solve(B: Tensor) -> Tensor:
        return FactorizedSolve.apply(B, chol_solver)

    return solve

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
    return factorize(A)(B)

def euler_bernoulli_energy(xs: Tensor, is_looped: bool = False, Ns: Optional[Tensor] = None):
    if is_looped:
        edges = diff(torch.cat([xs[..., -1:, :], xs, xs[..., :1, :]], dim=-2), dim=-2)
    else:
        edges = diff(xs, dim=-2)

    prev_edges = edges[..., :-1, :]
    next_edges = edges[..., 1:, :]
    
    if Ns is not None:
        if not is_looped:
            Ns = Ns[..., 1:-1, :]

        prev_edges = prev_edges - (Ns * prev_edges).sum(dim=-1, keepdims=True) * Ns
        next_edges = next_edges - (Ns * next_edges).sum(dim=-1, keepdims=True) * Ns

    prev_unit_edges = prev_edges / norm(prev_edges, dim=-1, keepdims=True)
    next_unit_edges = next_edges / norm(next_edges, dim=-1, keepdims=True)

    cos_turning_angles = (prev_unit_edges * next_unit_edges).sum(dim=-1)
    sin_turning_angles = norm(cross(prev_unit_edges, next_unit_edges, dim=-1), dim=-1)
    turning_angles = atan2(sin_turning_angles, cos_turning_angles)

    dual_lengths = (norm(prev_edges, dim=-1) + norm(next_edges, dim=-1)) / 2

    energy = ((turning_angles ** 2) / dual_lengths).sum(dim=-1) / 2

    if energy.isnan().any():
        assert False

    return energy

def pairwise_dists_to_pair_correlation_function(dists: Tensor, surface_area: float, sigma: float, rs: Tensor):
    g_rs = []
    for r in rs:
        K_r = exp(-((r - dists) ** 2) / (sigma ** 2)) / (sqrt(tensor(pi, dtype=dists.dtype)) * sigma)
        K_r -= diag(diag(K_r))
        g_r = K_r.sum() / r
        g_rs.append(g_r)

    g_rs = tensor(g_rs)
    g_rs *= surface_area / (2 * pi * (len(dists) ** 2))
    
    return g_rs