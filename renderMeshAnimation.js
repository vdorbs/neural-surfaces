function renderMeshAnimation(position_frames, indices, normal_frames, uvs, mode) {
    const canvas = document.getElementById("renderCanvas");
    const engine = new BABYLON.Engine(canvas, true);
    const scene = new BABYLON.Scene(engine);

    const mesh = new BABYLON.Mesh("mesh", scene);
    vertexData = new BABYLON.VertexData();
    vertexData.positions = position_frames[0];
    vertexData.indices = indices;
    vertexData.normals = normal_frames[0];
    vertexData.uvs = uvs
    vertexData.applyToMesh(mesh);

    mat = new BABYLON.StandardMaterial("mat", scene);
    mat.backFaceCulling = false;

    if (mode == "checkerboard") {
        mat.diffuseTexture = new BABYLON.Texture("https://i.imgur.com/g7C6P1m.png");
    } else if (mode == "turbo") {
        mat.diffuseTexture = new BABYLON.Texture("https://1.bp.blogspot.com/-T2q4LV_VaTA/XVWYfIwvOVI/AAAAAAAAEcQ/aUciAXkV_QAuuZ1y5DcbstBcDr-Umw4kgCLcBGAs/s1600/image10.png");
    } else if (mode == "none") {
        mat.diffuseColor = new BABYLON.Color3(0.678, 0.847, 0.902);
    };
    mesh.material = mat;

    scene.createDefaultCameraOrLight(true, true, true);
    scene.cameras[0].alpha = -Math.PI / 4;
    scene.cameras[0].beta = 1.25;

    extraLight = new BABYLON.HemisphericLight("extraLight", new BABYLON.Vector3(0, -1, 0), scene);
    scene.lights[0].intensity = 0.8;
    scene.lights[1].intensity = 0.3;

    env = scene.createDefaultEnvironment({enableGroundMirror: true});
    env.setMainColor(BABYLON.Color3.White());

    counter = 0
    scene.registerBeforeRender(function() {
        vertexData.positions = position_frames[counter];
        vertexData.normals = normal_frames[counter];
        vertexData.applyToMesh(mesh);
        counter = (counter + 1) % position_frames.length;
    });

    engine.runRenderLoop(function() {
        scene.render();
    });

    window.addEventListener("resize", function() {
        engine.resize();
    });
};