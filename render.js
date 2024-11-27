function renderMultiScene(objects, numFrames, frameLength) {
    const engineCanvas = document.getElementById("engineCanvas");
    const engine = new BABYLON.Engine(engineCanvas);

    const scenes = [];
    const allSceneCanvases = [];
    const rows = document.getElementsByClassName("row");
    for (const row of rows) {
        const sceneCanvases = row.getElementsByClassName("sceneCanvas");
        for (const sceneCanvas of sceneCanvases) {
            const scene = new BABYLON.Scene(engine);
            scene.useRightHandedSystem = true;
            scenes.push(scene);
            allSceneCanvases.push(sceneCanvas);
        };
    };

    const animationFrameCounters = [];
    const animatedObjectsBySceneId = [];
    for (const scene of scenes) {
        animationFrameCounters.push(0);
        animatedObjectsBySceneId.push([]);
    };

    for (let i = 0; i < objects.length; i++) {
        const object = objects[i];
        const scene = scenes[object.sceneId];
        if (object.isAnimated) {
            animatedObjectsBySceneId[object.sceneId].push(i);
        };

        if (object.type == "mesh") {
            const mesh = new BABYLON.Mesh("mesh" + i, scene);
            const material = new BABYLON.StandardMaterial("meshMat" + i, scene);
            const vertexData = new BABYLON.VertexData();

            if (object.hasUvs) {
                if (object.wrapUs) {
                    material.diffuseTexture = new BABYLON.Texture("https://i.imgur.com/dIzJUrT.png");
                } else {
                    material.diffuseTexture = new BABYLON.Texture("https://i.imgur.com/g7C6P1m.png");
                };
                // material.diffuseTexture.uScale = -1;
            } else if (object.hasColors) {
                vertexData.colors = object.isAnimated ? object.colors[0] : object.colors;
            } else {
                material.diffuseColor = new BABYLON.Color3(0.678, 0.847, 0.902);
            };
            
            material.backFaceCulling = false;
            mesh.material = material;

            vertexData.positions = object.isAnimated ? object.positions[0] : object.positions;
            vertexData.indices = object.isAnimated ? object.indices[0] : object.indices;
            vertexData.normals = object.isAnimated ? object.normals[0] : object.normals;

            if (object.hasUvs) {
                vertexData.uvs = object.isAnimated ? object.uvs[0] : object.uvs;
            };

            vertexData.applyToMesh(mesh);
        };

        if (object.type == "pointCloud") {
            const diameter = 2 * object.radii;
            for (let j = 0; j < object.numPoints; j++) {
                const position = object.isAnimated ? object.positions[0][j] : object.positions[j];
                const positionVector = new BABYLON.Vector3(position[0], position[1], position[2]);
                const sphere = BABYLON.MeshBuilder.CreateSphere("pointCloud" + i + "Sphere" + j, {diameter: diameter}, scene);
                sphere.position = positionVector;

                if (object.hasColors) {
                    const material = new BABYLON.StandardMaterial("pointCloud" + i + "Sphere" + j + "Mat", scene);
                    const color = object.isAnimated ? object.colors[0][j]: object.colors[j];
                    material.diffuseColor = new BABYLON.Color3(color[0], color[1], color[2]);
                    sphere.material = material;
                };
            };
        };

        if (object.type == "curve") {
            const positionVectors = [];
            for (const position of object.positions) {
                positionVectors.push(new BABYLON.Vector3(position[0], position[1], position[2]));
            };

            if (object.isLooped) {
                positionVectors.push(positionVectors[0]);
                positionVectors.push(positionVectors[1]);
            };

            const curve = BABYLON.MeshBuilder.CreateTube("curve" + i, {path: positionVectors, radius: object.radius, sideOrientation: BABYLON.Mesh.FRONTSIDE, cap: BABYLON.Mesh.CAP_ALL}, scene);
            if (object.hasColors) {
                const material = new BABYLON.StandardMaterial("curve" + i + "Mat", scene);
                const color = object.colors
                material.diffuseColor = new BABYLON.Color3(color[0], color[1], color[2]);
                curve.material = material;
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

        let animatedObjects = animatedObjectsBySceneId[i];
        if (animatedObjects.length > 0) {
            scene.registerBeforeRender(function() {
                if (animationFrameCounters[i] % frameLength == 0) {
                    const effectiveFrame = Math.floor(animationFrameCounters[i] / frameLength);
                    for (let j = 0; j < animatedObjects.length; j++) {
                        const objectId = animatedObjects[j];
                        const object = objects[objectId];

                        if (object.type == "mesh") {
                            const mesh = scene.getMeshByName("mesh" + objectId);
                            const vertexData = new BABYLON.VertexData();
                            vertexData.positions = object.positions[effectiveFrame];
                            vertexData.indices = object.indices[effectiveFrame];
                            vertexData.normals = object.normals[effectiveFrame];

                            if (object.hasUvs) {
                                vertexData.uvs = object.uvs[effectiveFrame];
                            } else if (object.hasColors) {
                                vertexData.color = object.colors[effectiveFrame];
                            };

                            console.log(i + " mesh " + objectId)
                            vertexData.applyToMesh(mesh);
                        };

                        if (object.type == "pointCloud") {
                            for (let k = 0; k < object.numPoints; k++) {
                                const sphere = scene.getMeshByName("pointCloud" + objectId + "Sphere" + k);
                                const position = object.positions[effectiveFrame][k];
                                const positionVector = new BABYLON.Vector3(position[0], position[1], position[2]);
                                sphere.position = positionVector;

                                if (object.hasColors) {
                                    const color = object.colors[effectiveFrame][k];
                                    sphere.material.diffuseColor = new BABYLON.Color3(color[0], color[1], color[2]);
                                };
                            };
                        };

                    };
                };
                animationFrameCounters[i] = (animationFrameCounters[i] + 1) % (numFrames * frameLength);
            });
        };
        
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