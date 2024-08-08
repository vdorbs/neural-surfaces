def render_meshes(all_fs, all_faces, all_uvs):
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
