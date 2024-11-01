function renderMultiScene(objects) {
    const engineCanvas = document.getElementById("engineCanvas");
    const engine = new BABYLON.Engine(engineCanvas);

    const scenes = [];
    const allSceneCanvases = [];
    const rows = document.getElementsByClassName("row");
    for (const row of rows) {
        const sceneCanvases = row.getElementsByClassName("sceneCanvas");
        for (const sceneCanvas of sceneCanvases) {
            const scene = new BABYLON.Scene(engine);
            scenes.push(scene);
            allSceneCanvases.push(sceneCanvas);
        };
    };

    let meshId = 0;
    let pointCloudId = 0;
    for (let i = 0; i < objects.length; i++) {
        const object = objects[i];
        const scene = scenes[object.sceneId];

        if (object.type == "mesh") {
            const mesh = new BABYLON.Mesh("mesh" + meshId, scene);
            const material = new BABYLON.StandardMaterial("meshMat" + meshId, scene);
            const vertexData = new BABYLON.VertexData();

            if (object.hasUvs) {
                if (object.wrapUs) {
                    material.diffuseTexture = new BABYLON.Texture("https://i.imgur.com/dIzJUrT.png");
                } else {
                    material.diffuseTexture = new BABYLON.Texture("https://i.imgur.com/g7C6P1m.png");
                };
                material.diffuseTexture.uScale = -1;
            } else if (object.hasColors) {
                vertexData.colors = object.colors;
            } else {
                material.diffuseColor = new BABYLON.Color3(0.678, 0.847, 0.902);
            };
            
            material.backFaceCulling = false;
            mesh.material = material;

            vertexData.positions = object.positions;
            vertexData.indices = object.indices;

            if (object.hasUvs) {
                vertexData.uvs = object.uvs;
            };

            vertexData.applyToMesh(mesh);
            meshId += 1;
        };

        if (object.type == "pointCloud") {
            const diameter = 2 * object.radii;
            for (let j = 0; j < object.positions.length; j++) {
                const position = object.positions[j];
                const positionVector = new BABYLON.Vector3(position[0], position[1], position[2]);
                const sphere = BABYLON.MeshBuilder.CreateSphere("pointCloud" + pointCloudId + "Sphere" + j, {diameter: diameter}, scene);
                sphere.position = positionVector;

                if (object.hasColors) {
                    const material = new BABYLON.StandardMaterial("pointCloud" + pointCloudId + "Sphere" + j + "Mat", scene);
                    const color = object.colors[j];
                    material.diffuseColor = new BABYLON.Color3(color[0], color[1], color[2]);
                    sphere.material = material;
                };
            };
        };
    };

    const views = [];
    for (let i = 0; i < scenes.length; i++) {
        const scene = scenes[i];
        
        scene.createDefaultCameraOrLight(true, false, true);
        scene.cameras[0].alpha = -Math.PI / 4;
        scene.cameras[0].beta = 1.25;

        const extraLight = new BABYLON.HemisphericLight("extraLight" + i, new BABYLON.Vector3(0, -1, 0), scene);
        scene.lights[0].intensity = 0.8;
        extraLight.intensity = 0.5;

        env = scene.createDefaultEnvironment({enableGroundMirror: true});
        env.setMainColor(BABYLON.Color3.White());

        const view = engine.registerView(allSceneCanvases[i], scene.cameras[0]);
        views.push(view);
    };

    engine.inputElement = allSceneCanvases[0];

    engine.runRenderLoop(function() {
        for (let i = 0; i < views.length; i++) {
            if (views[i] === engine.activeView) {
                scenes[i].render();
            }
        };
    });
};