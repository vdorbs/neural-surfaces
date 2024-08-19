function addEventListeners(engine, canvas, scene) {
    canvas.addEventListener("mouseover", function() {
        scene.attachControl();
        engine.inputElement = canvas;
    });

    canvas.addEventListener("mouseout", function() {
        scene.detachControl();
    });
};

function renderMeshes(all_positions, all_indices, all_uvs, mode) {
    const engineCanvas = document.getElementById("engineCanvas");
    const engine = new BABYLON.Engine(engineCanvas, true);

    var texture_url;
    if (mode == "checkerboard") {
        texture_url = "https://us.v-cdn.net/5021068/uploads/editor/ha/7frj09nru4zu.png";
    } else if (mode == "turbo") {
        texture_url = "https://1.bp.blogspot.com/-T2q4LV_VaTA/XVWYfIwvOVI/AAAAAAAAEcQ/aUciAXkV_QAuuZ1y5DcbstBcDr-Umw4kgCLcBGAs/s1600/image10.png";
    };

    const rows = document.getElementsByClassName("row");
    var views = [];
    var i, j, canvases, scene, vertexData, view, env;
    for (i = 0; i < rows.length; i++) {
        canvases = rows[i].getElementsByTagName("canvas");
        for (j = 0; j < canvases.length; j++) {
            scene = new BABYLON.Scene(engine);
            scene.detachControl();

            mesh = new BABYLON.Mesh("mesh", scene);
            vertexData = new BABYLON.VertexData();
            vertexData.positions = all_positions[i][j];
            vertexData.indices = all_indices[i][j]
            vertexData.uvs = all_uvs[i][j];
            vertexData.applyToMesh(mesh);

            mat = new BABYLON.StandardMaterial("mat", scene);
            mat.diffuseTexture = new BABYLON.Texture(texture_url);
            mat.backFaceCulling = false;
            mesh.material = mat;
            
            scene.createDefaultCameraOrLight(true, true, true);
            scene.cameras[0].alpha = -Math.PI / 4;
            scene.cameras[0].beta = 1.25;
            view = engine.registerView(canvases[j], scene.cameras[0]);
            views.push(view);

            env = scene.createDefaultEnvironment({enableGroundMirror: true});
            env.setMainColor(BABYLON.Color3.White());

            addEventListeners(engine, canvases[j], scene);
        };
    };

    engine.runRenderLoop(function() {
        for (var i = 0; i < views.length; i++) {
            if (engine.activeView == views[i]) {
                engine.scenes[i].render();
            };
        };
    });
};