function renderPointCloudAnimation(frames, positions, indices, framesPerUpdate) {
    const canvas = document.getElementById("renderCanvas");
    const engine = new BABYLON.Engine(canvas, true);
    const scene = new BABYLON.Scene(engine);

    const mesh = new BABYLON.Mesh("mesh", scene);
    vertexData = new BABYLON.VertexData();
    vertexData.positions = positions;
    vertexData.indices = indices;
    vertexData.applyToMesh(mesh);

    const mat = new BABYLON.StandardMaterial("mat", scene);
    mat.backFaceCulling = false;
    mat.diffuseColor = new BABYLON.Color3(0.678, 0.847, 0.902);
    mesh.material = mat;

    const pointCloud = new BABYLON.PointsCloudSystem("pointCloud", 10, scene);
    pointCloud.addPoints(frames[0].length, function(particle, i) {
        const position = frames[0][i];
        particle.position = new BABYLON.Vector3(position[0], position[1], position[2])
        particle.color = new BABYLON.Color3(1, 0.843, 0);
    });
    pointCloud.buildMeshAsync()

    pointCloud.updateParticle = function(particle) {
        const position = frames[pointCloud.counter][particle.idx];
        particle.position.x = position[0];
        particle.position.y = position[1];
        particle.position.z = position[2];
    };

    var step = 0;
    pointCloud.afterUpdateParticles = function() {
        step = (step + 1) % framesPerUpdate;
        if (step === 0) {
            pointCloud.counter = (pointCloud.counter + 1) % frames.length;
        };
    };

    scene.registerBeforeRender(function() {
        pointCloud.setParticles(0, frames[0].length, true);
    });

    scene.createDefaultCameraOrLight(true, true, true);
    scene.cameras[0].alpha = -Math.PI / 4;
    scene.cameras[0].beta = 1.25;

    const extraLight = new BABYLON.HemisphericLight("extraLight", new BABYLON.Vector3(0, -1, 0), scene);
    scene.lights[0].intensity = 0.8;
    extraLight.intensity = 0.3;

    const env = scene.createDefaultEnvironment({enableGroundMirror: true});
    env.setMainColor(BABYLON.Color3.White());

    engine.runRenderLoop(function() {
        scene.render();
    });

    window.addEventListener("resize", function() {
        engine.resize();
    });
};