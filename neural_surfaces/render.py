from matplotlib.cm import turbo
from torch import arange, cat, diff, pi, stack, Tensor, tensor
from typing import Optional


class MultiScene:
    """Render with Babylon.js a grid of scenes with synchronized cameras, containing meshes and/or point clouds, with camera control attached to the upper leftmost scene"""

    def __init__(self, num_rows: int, num_cols: int, alpha: float = -pi / 4, beta = 1.25, num_frames: int = -1, frame_length: int = -1):
        """
        Args:
            num_rows: number of rows
            num_cols: number of columns
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.alpha = alpha
        self.beta = beta
        self.num_frames = num_frames
        self.frame_length = frame_length

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
            <script src="https://cdn.babylonjs.com/gui/babylon.gui.js"></script>
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

    def add_mesh(self, row: int, col: int, fs: Tensor, faces: Tensor, Ns: Tensor, uvs: Optional[Tensor] = None, wrap_us: bool = False, cs: Optional[Tensor] = None, y_up: bool = False, is_animated: bool = False):
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
            is_animated (bool): whether or not data is dynamic, to be rendered as an animation
        """
        scene_id = row * self.num_cols + col
        if not y_up:
            fs = fs[..., tensor([1, 2, 0])]

        if not is_animated:
            fs = fs.unsqueeze(0)
            Ns = Ns.unsqueeze(0)
            
            if uvs is not None:
                uvs = uvs.unsqueeze(0)

            if cs is not None:
                cs = cs.unsqueeze(0)

        all_positions = []
        all_indices = []
        all_normals = []

        if uvs is not None:
            all_uvs = []

        if cs is not None:
            all_colors = []

        for frame in range(len(fs)):
            if uvs is not None and wrap_us:
                us_by_face = uvs[frame, :, 0][faces]
                crosses_seam = (diff(us_by_face, dim=-1).abs() > 0.75).any(dim=-1)
                crossing_faces = faces[crosses_seam]
                crossing_fs = fs[frame, crossing_faces].flatten(end_dim=-2)
                crossing_Ns = Ns[frame, crossing_faces].flatten(end_dim=-2)

                crossing_us_by_face = us_by_face[crosses_seam]
                crossing_us_by_face += (crossing_us_by_face < 0.5)
                crossing_us = crossing_us_by_face.flatten()
                crossing_vs = uvs[frame, :, 1][crossing_faces].flatten()
                crossing_uvs = stack([crossing_us, crossing_vs], dim=-1)

                all_positions.append(cat([fs[frame], crossing_fs]).flatten().tolist())
                all_normals.append(cat([Ns[frame], crossing_Ns]).flatten().tolist())
                wrapped_uvs = cat([uvs[frame], crossing_uvs])
                wrapped_uvs /= tensor([[2, 1]])
                wrapped_uvs = wrapped_uvs.flatten().tolist()
                all_uvs.append(wrapped_uvs)

                crossing_faces = faces.max() + 1 + arange(len(crossing_fs)).reshape(-1, 3)
                all_indices.append(cat([faces[~crosses_seam], crossing_faces]).flatten().tolist())

            else:
                all_positions.append(fs[frame].flatten().tolist())
                all_indices.append(faces.flatten().tolist())
                all_normals.append(Ns[frame].flatten().tolist())

                if uvs is not None:
                    all_uvs.append(uvs[frame].flatten().tolist())

                elif cs is not None:
                    all_colors.append(turbo(cs[frame]).flatten().tolist())

        if not is_animated:
            all_positions = all_positions[0]
            all_indices = all_indices[0]
            all_normals = all_normals[0]

            if uvs is not None:
                all_uvs = all_uvs[0]
            
            elif cs is not None:
                all_colors = all_colors[0]

        has_uvs = 'true' if uvs is not None else 'false'
        has_colors = 'true' if uvs is None and cs is not None else 'false'
        is_animated = 'true' if is_animated else 'false'
        obj_str = f"""{{ type: "mesh", sceneId: {scene_id}, positions: {all_positions}, indices: {all_indices}, normals: {all_normals}, hasUvs: {has_uvs}, hasColors: {has_colors}, isAnimated: {is_animated}"""
        
        if uvs is not None:
            wrap_us = 'true' if wrap_us else 'false'
            obj_str = f"""{obj_str}, uvs: {all_uvs}, wrapUs: {wrap_us}}}"""
        elif cs is not None:
            obj_str = f"""{obj_str}, colors: {all_colors}}}"""
        else:
            obj_str = f"""{obj_str}}}"""

        self.obj_strs.append(obj_str)

    def add_point_cloud(self, row: int, col: int, xs: Tensor, radii: float = 0.1, cs: Optional[Tensor] = None, y_up: bool = False, is_animated: bool = False):
        """Adds a point cloud to a specified scene, with no colors or colors from the Turbo colormap

        Args:
            row (int): zero-indexed row
            col (int): zero-indexed column
            xs (Tensor): num_points * 3 list of point positions
            radii (float): radii of spheres at points
            cs (Optional[Tensor]): num_points list of colors from Turbo colormap, in range 0 to 1
            y_up (bool): whether x points right, y points up, z points forward or x points forward, y points right, z points up
            is_animated (bool): whether or not data is dynamic, to be rendered as an animation
        """
        scene_id = row * self.num_cols + col
        if not y_up:
            xs = xs[..., tensor([1, 2, 0])]

        positions = xs.tolist()
        has_colors = 'true' if cs is not None else 'false'
        is_animated = 'true' if is_animated else 'false'
        obj_str = f"""{{ type: "pointCloud", sceneId: {scene_id}, numPoints: {xs.shape[-2]}, positions: {positions}, radii: {radii}, hasColors: {has_colors}, isAnimated: {is_animated}"""

        if cs is not None:
            colors = turbo(cs)[..., :3].tolist()
            obj_str = f"""{obj_str}, colors: {colors}}}"""
        else:
            obj_str = f"""{obj_str}}}"""

        self.obj_strs.append(obj_str)

    def add_curve(self, row: int, col: int, xs: Tensor, is_looped: bool = False, radius: float = 0.1, color: Optional[Tensor] = None, y_up: bool = False, is_animated: bool = False):
        """Adds a curve to a specified scene, with no colors or colors from the Turbo colormap

        Args:
            row (int): zero-indexed row
            col (int): zero-indexed column
            xs (Tensor): num_points * 3 list of curve vertex positions
            is_looped (bool): whether or not curve is a loop
            radius (float): radius of tube around curve
            color (Optional[Tensor]): color from Turbo colormap, in range 0 to 1
            y_up (bool): whether x points right, y points up, z points forward or x points forward, y points right, z points up
            is_animated (bool): whether or not data is dynamic, to be rendered as an animation
        """
        scene_id = row * self.num_cols + col
        if not y_up:
            xs = xs[..., tensor([1, 2, 0])]

        if is_looped:
            xs = cat([xs, xs[..., 0:2, :]], dim=-2)

        positions = xs.tolist()
        has_colors = 'true' if color is not None else 'false'
        is_animated = 'true' if is_animated else 'false'
        obj_str = f"""{{ type: "curve", sceneId: {scene_id}, positions: {positions}, radius: {radius}, hasColors: {has_colors}, isAnimated: {is_animated}"""
        
        if color is not None:
            if is_animated == 'true':
                color = turbo(color)[:, :3].tolist()
            else:
                color = tensor(turbo(color)[:3]).tolist()

            obj_str = f"""{obj_str}, colors: {color}}}"""
        else:
            obj_str = f"""{obj_str}}}"""

        self.obj_strs.append(obj_str)

    def make(self) -> str:
        """Generate HTML string for rendering"""
        js_str = ", ".join(self.obj_strs)
        js_str = f'renderMultiScene([{js_str}], {self.alpha}, {self.beta}, {self.num_frames}, {self.frame_length})'
        return self.pre_html_str + js_str + self.post_html_str
