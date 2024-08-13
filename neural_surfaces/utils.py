from http.server import SimpleHTTPRequestHandler
from io import BytesIO
from socketserver import TCPServer
from torch import Tensor, tensor
from trimesh.exchange.obj import load_obj
from typing import List, Tuple
from urllib.request import urlopen


BUNNY_URL = 'https://raw.githubusercontent.com/odedstein/meshes/master/objects/bunny/bunny.obj'
CAT_URL = 'https://raw.githubusercontent.com/odedstein/meshes/master/objects/cat/cat-low-resolution.obj'
PENGUIN_URL = 'https://raw.githubusercontent.com/odedstein/meshes/master/objects/penguin/penguin.obj'
SPOT_URL = 'https://raw.githubusercontent.com/odedstein/meshes/master/objects/spot/spot_low_resolution.obj'

def load_obj_from_url(url: str) -> Tuple[Tensor, Tensor]:
    """Loads mesh data from obj file at url

    Returns:
        num_vertices * 3 list of vertex positions and num_faces * 3 list of vertices per face
    """
    with urlopen(url) as response:
        buffer = BytesIO(response.read())
    mesh = load_obj(buffer, maintain_order=True)
    return tensor(mesh['vertices']), tensor(mesh['faces'])

def meshes_to_html(all_fs: List[List[Tensor]], all_faces: List[List[Tensor]], all_uvs: List[List[Tensor]]) -> str:
    """Creates HTML string for rendering textured meshes with Babylon.js

    Note:
        num_vertices and num_faces can be different for each mesh

    Args:
        all_fs (List[List[Tensor]]): num_rows list of num_cols lists of num_vertices * 3 lists of vertex positions
        all_faces (List[List[Tensor]]): num_rows list of num_cols lists of num_faces * 3 lists of vertices per face
        all_uvs (List[List[Tensor]]): num_rows list of num_cols lists of num_vertices * 2 lists of uv coordinates per vertex

    Returns:
        HTML string, can be saved to a file or logged to a HTML-supported logger
    """

    num_rows = len(all_fs)
    num_cols = len(all_fs[0])

    all_positions = [[fs.flatten().tolist() for fs in fs_row] for fs_row in all_fs]
    all_indices = [[faces.flatten().tolist() for faces in faces_row] for faces_row in all_faces]
    all_uvs = [[uvs.flatten().tolist() for uvs in uvs_row] for uvs_row in all_uvs]

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
                        renderMeshes({all_positions}, {all_indices}, {all_uvs});
                    </script>
                </body>
                </html>
                """
    
    return html_str

def serve_html(html_str: str, port: int = 8000):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_str.encode())

    with TCPServer(('', port), Handler) as httpd:
        print(f'Serving at https://localhost:{port}')
        httpd.serve_forever()
