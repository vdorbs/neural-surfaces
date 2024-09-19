function renderPointCloudAnimation(frames, radii, frameRate, positions, indices, normals) {
    const canvas = document.getElementById("renderCanvas");
    const engine = new BABYLON.Engine(canvas, true);
    const scene = new BABYLON.Scene(engine);

    const mesh = new BABYLON.Mesh("mesh", scene);
    vertexData = new BABYLON.VertexData();
    vertexData.positions = positions;
    vertexData.indices = indices;
    vertexData.normals = normals;
    vertexData.applyToMesh(mesh);

    mesh_mat = new BABYLON.StandardMaterial("mesh_mat", scene);
    mesh_mat.backFaceCulling = false;
    mesh_mat.diffuseColor = new BABYLON.Color3(0.678, 0.847, 0.902);
    mesh_mat.alpha = 0.5;
    mesh.material = mesh_mat;

    allKeyFrames = [];
    spheres = [];
    for (j = 0; j < frames[0].length; j++) {
        allKeyFrames.push([]);
        sphere = BABYLON.MeshBuilder.CreateSphere("sphere_" + j, {diameter: 2 * radii});
        spheres.push(sphere);
    };

    for (i = 0; i < frames.length; i++) {
        frame = frames[i];
        for (j = 0; j < frame.length; j++) {
            position = frame[j];
            vector = new BABYLON.Vector3(position[0], position[1], position[2]);
            allKeyFrames[j].push({frame: i, value: vector});
        };
    };

    for (j = 0; j < frames[0].length; j++) {
        anim = new BABYLON.Animation("anim_" + j, "position", frameRate, BABYLON.Animation.ANIMATIONTYPE_VECTOR3, BABYLON.Animation.ANIMATIONLOOPMODE_CYCLE);
        anim.setKeys(allKeyFrames[j]);
        sphere = spheres[j];
        sphere.animations.push(anim);
        scene.beginAnimation(sphere, 0, frames.length / frameRate, true);
    };

    scene.createDefaultCameraOrLight(true, true, true);
    scene.cameras[0].alpha = -Math.PI / 4;
    scene.cameras[0].beta = 1.25;

    extraLight = new BABYLON.HemisphericLight("extraLight", new BABYLON.Vector3(0, -1, 0), scene);
    scene.lights[0].intensity = 0.8;
    scene.lights[1].intensity = 0.3;

    env = scene.createDefaultEnvironment({enableGroundMirror: true});
    env.setMainColor(BABYLON.Color3.White());

    engine.runRenderLoop(function() {
        scene.render();
    });

    window.addEventListener("resize", function() {
        engine.resize();
    });
};