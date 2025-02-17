function addEventListeners(engine, canvas, scene) {
    canvas.addEventListener("mouseover", function() {
        scene.attachControl();
        engine.inputElement = canvas;
    });

    canvas.addEventListener("mouseout", function() {
        scene.detachControl();
    });
};

function renderMeshes(all_positions, all_indices, all_normals, all_uvs, mode) {
    const engineCanvas = document.getElementById("engineCanvas");
    const engine = new BABYLON.Engine(engineCanvas, true);

    const rows = document.getElementsByClassName("row");
    var views = [];
    var i, j, canvases, scene, vertexData, view, env, extraLight;
    for (i = 0; i < rows.length; i++) {
        canvases = rows[i].getElementsByTagName("canvas");
        for (j = 0; j < canvases.length; j++) {
            scene = new BABYLON.Scene(engine);
            scene.detachControl();

            mesh = new BABYLON.Mesh("mesh", scene);
            vertexData = new BABYLON.VertexData();
            vertexData.positions = all_positions[i][j];
            vertexData.indices = all_indices[i][j];
            vertexData.normals = all_normals[i][j];
            vertexData.uvs = all_uvs[i][j];
            vertexData.applyToMesh(mesh);

            mat = new BABYLON.StandardMaterial("mat", scene);
            mat.backFaceCulling = false;

            if (mode == "checkerboard") {
                mat.diffuseTexture = new BABYLON.Texture("https://i.imgur.com/dIzJUrT.png");
            } else if (mode == "turbo") {
                mat.diffuseTexture = new BABYLON.Texture("https://1.bp.blogspot.com/-T2q4LV_VaTA/XVWYfIwvOVI/AAAAAAAAEcQ/aUciAXkV_QAuuZ1y5DcbstBcDr-Umw4kgCLcBGAs/s1600/image10.png");
            } else if (mode == "none") {
                mat.diffuseColor = new BABYLON.Color3(0.678, 0.847, 0.902);
            };
            mesh.material = mat;
            
            scene.createDefaultCameraOrLight(true, true, true);
            scene.cameras[0].alpha = -Math.PI / 4;
            scene.cameras[0].beta = 1.25;
            view = engine.registerView(canvases[j], scene.cameras[0]);
            views.push(view);

            extraLight = new BABYLON.HemisphericLight("extraLight", new BABYLON.Vector3(0, -1, 0), scene);
            scene.lights[0].intensity = 0.8;
            scene.lights[1].intensity = 0.3;

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