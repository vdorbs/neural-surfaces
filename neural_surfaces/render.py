from matplotlib.cm import turbo
from torch import arange, cat, diff, stack, Tensor, tensor
from typing import Optional


class MultiScene:
    """Render with Babylon.js a grid of scenes with synchronized cameras, containing meshes and/or point clouds, with camera control attached to the upper leftmost scene"""

    def __init__(self, num_rows: int, num_cols: int):
        """
        Args:
            num_rows: number of rows
            num_cols: number of columns
        """

        self.num_rows = num_rows
        self.num_cols = num_cols

        row_str = """<canvas class="sceneCanvas"></canvas>""" * num_cols
        row_str = f"""<div class="row">{row_str}</div>"""
        body_str = row_str * num_rows
        body_str = f"""<canvas id="engineCanvas"></canvas>{body_str}"""

        with open('render.js') as f:
            js_str = f.read()

        self.pre_html_str = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                canvas#engineCanvas {{ width: 0; height: 0 }}
                div.row {{ display: flex }}
                canvas.sceneCanvas {{ width: {100 // num_cols}vw; height: {100 // num_rows}vh }}
            </style>
            <script src="https://cdn.babylonjs.com/babylon.js"></script>
        </head>
        <body>
            {body_str}
            <script>
                {js_str}
        """

        self.obj_strs = []

        self.post_html_str = f"""
            </script>
        </body>
        """

    def add_mesh(self, row: int, col: int, fs: Tensor, faces: Tensor, Ns: Tensor, uvs: Optional[Tensor] = None, wrap_us: bool = False, cs: Optional[Tensor] = None, y_up: bool = False):
        """Adds a mesh to a specified scene, with no colors, colors from a checkerboard pattern, or colors from the Turbo colormap
        
        Note:
            If both uvs and cs are provided, cs will be ignored

        Args:
            row (int): zero-indexed row
            col (int): zero-indexed column
            fs (Tensor): num_vertices * 3 list of vertex positions
            faces (Tensor): num_faces * 3 list of vertices per face
            Ns (Tensor): num_vertices * 3 list of vertex normals
            uvs (Optional[Tensor]): num_vertices * 2 list of UV coordinates per vertex, in range 0 to 1
            wrap_us (bool): whether or not U coordinates should wrap around a seam
            cs (Optional[Tensor]): num_vertices list of colors from Turbo colormap, in range 0 to 1
            y_up (bool): whether x points right, y points up, z points forward or x points forward, y points right, z points up
        """
        scene_id = row * self.num_cols + col
        if not y_up:
            fs = fs[:, tensor([1, 2, 0])]

        if uvs is not None and wrap_us:
            us_by_face = uvs[:, 0][faces]
            crosses_seam = (diff(us_by_face, dim=-1).abs() > 0.75).any(dim=-1)
            crossing_faces = faces[crosses_seam]
            crossing_fs = fs[crossing_faces].flatten(end_dim=-2)
            crossing_Ns = Ns[crossing_faces].flatten(end_dim=-2)

            crossing_us_by_face = us_by_face[crosses_seam]
            crossing_us_by_face += (crossing_us_by_face < 0.5)
            crossing_us = crossing_us_by_face.flatten()
            crossing_vs = uvs[:, 1][crossing_faces].flatten()
            crossing_uvs = stack([crossing_us, crossing_vs], dim=-1)

            positions = cat([fs, crossing_fs]).flatten().tolist()
            normals = cat([Ns, crossing_Ns]).flatten().tolist()
            uvs = cat([uvs, crossing_uvs])
            uvs /= tensor([[2, 1]])
            uvs = uvs.flatten().tolist()

            crossing_faces = faces.max() + 1 + arange(3 * len(crossing_fs)).reshape(-1, 3)
            indices = cat([faces[~crosses_seam], crossing_faces]).flatten().tolist()

        else:
            positions = fs.flatten().tolist()
            indices = faces.flatten().tolist()
            normals = Ns.flatten().tolist()

            if uvs is not None:
                uvs = uvs.flatten().tolist()

            elif cs is not None:
                colors = turbo(cs).flatten().tolist()

        has_uvs = 'true' if uvs is not None else 'false'
        has_colors = 'true' if uvs is None and cs is not None else 'false'
        obj_str = f"""{{ type: "mesh", sceneId: {scene_id}, positions: {positions}, indices: {indices}, normals: {normals}, hasUvs: {has_uvs}, hasColors: {has_colors}"""
        
        if uvs is not None:
            wrap_us = 'true' if wrap_us else 'false'
            obj_str = f"""{obj_str}, uvs: {uvs}, wrapUs: {wrap_us}}}"""
        elif cs is not None:
            obj_str = f"""{obj_str}, colors: {colors}}}"""
        else:
            obj_str = f"""{obj_str}}}"""

        self.obj_strs.append(obj_str)

    def add_point_cloud(self, row: int, col: int, xs: Tensor, radii: float = 0.1, cs: Optional[Tensor] = None, y_up: bool = False):
        """Adds a point cloud to a specified scene, with no colors or colors from the Turbo colormap

        Args:
            row (int): zero-indexed row
            col (int): zero-indexed column
            xs (Tensor): num_points * 3 list of point positions
            radii (float): radii of spheres at points
            cs (Optional[Tensor]): num_points list of colors from Turbo colormap, in range 0 to 1
            y_up (bool): whether x points right, y points up, z points forward or x points forward, y points right, z points up
        """
        scene_id = row * self.num_cols + col
        if not y_up:
            xs = xs[:, tensor([1, 2, 0])]

        positions = xs.tolist()
        has_colors = 'true' if cs is not None else 'false'
        obj_str = f"""{{ type: "pointCloud", sceneId: {scene_id}, positions: {positions}, radii: {radii}, hasColors: {has_colors}"""

        if cs is not None:
            colors = turbo(cs)[:, :3].tolist()
            obj_str = f"""{obj_str}, colors: {colors}}}"""
        else:
            obj_str = f"""{obj_str}}}"""

        self.obj_strs.append(obj_str)

    def make(self) -> str:
        """Generate HTML string for rendering"""
        js_str = ", ".join(self.obj_strs)
        js_str = f'renderMultiScene([{js_str}])'
        return self.pre_html_str + js_str + self.post_html_str
