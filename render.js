function renderMultiScene(objects, alpha, beta, numFrames, frameLength) {
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
                    material.diffuseTexture = new BABYLON.Texture("https://i.imgur.com/DMYF8Q7.png");
                } else {
                    material.diffuseTexture = new BABYLON.Texture("https://i.imgur.com/zXCNI7h.png");
                };
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
                const sphere = BABYLON.MeshBuilder.CreateSphere("pointCloud" + i + "Sphere" + j, {diameter: diameter, segments: 8}, scene);
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
            positions = object.isAnimated ? object.positions[0] : object.positions;
            const positionVectors = [];
            for (const position of positions) {
                positionVectors.push(new BABYLON.Vector3(position[0], position[1], position[2]));
            };

            const curve = BABYLON.MeshBuilder.CreateTube("curve" + i, {path: positionVectors, radius: object.radius, sideOrientation: BABYLON.Mesh.FRONTSIDE, cap: BABYLON.Mesh.CAP_ALL, updatable: true}, scene);
            if (object.hasColors) {
                const material = new BABYLON.StandardMaterial("curve" + i + "Mat", scene);
                const color = object.isAnimated ? object.colors[0] : object.colors;
                material.diffuseColor = new BABYLON.Color3(color[0], color[1], color[2]);
                curve.material = material;
            };
        };
    };

    const views = [];
    for (let i = 0; i < scenes.length; i++) {
        const scene = scenes[i];
        
        scene.createDefaultCameraOrLight(true, false, true);
        scene.cameras[0].alpha = alpha;
        scene.cameras[0].beta = beta;

        const extraLight = new BABYLON.HemisphericLight("extraLight" + i, new BABYLON.Vector3(0, -1, 0), scene);
        const shadowLight = new BABYLON.DirectionalLight("shadowLight", new BABYLON.Vector3(-1, -1.5, 1), scene);
        scene.lights[0].intensity = 0.8;
        extraLight.intensity = 0.5;
        shadowLight.intensity = 0.5;

        env = scene.createDefaultEnvironment();
        env.setMainColor(new BABYLON.Color3(10, 10, 10));
        env.ground.receiveShadows = true;

        const generator = new BABYLON.ShadowGenerator(2048, shadowLight);
        generator.usePercentageCloserFiltering = true;
        generator._darkness = -0.75;

        for (let j = 0; j < objects.length; j++) {
            if (objects[j].sceneId == i && objects[j].type == "mesh") {
                mesh = scene.getMeshByName("mesh" + j);
                generator.getShadowMap().renderList.push(mesh);
            };
        };

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
                                vertexData.colors = object.colors[effectiveFrame];
                            };

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

                        if (object.type == "curve") {
                            let tube = scene.getMeshByName("curve" + objectId);
                            const positions = object.positions[effectiveFrame]
                            
                            const positionVectors = [];
                            for (k = 0; k < positions.length; k++) {
                                const position = positions[k];
                                const positionVector = new BABYLON.Vector3(position[0], position[1], position[2]);
                                positionVectors.push(positionVector);
                            };

                            BABYLON.MeshBuilder.CreateTube("curve" + objectId, {path: positionVectors, instance: tube}, scene);

                            if (object.hasColors) {
                                const color = object.colors[effectiveFrame];
                                tube.material.diffuseColor = new BABYLON.Color3(color[0], color[1], color[2]);
                            };
                        };
                    };
                };
                animationFrameCounters[i] = (animationFrameCounters[i] + 1) % (numFrames * frameLength);
            });
        };
        
    };

    engine.inputElement = allSceneCanvases[0];

    const advancedTexture = new BABYLON.GUI.AdvancedDynamicTexture.CreateFullscreenUI("UI", true, scenes[0]);
    
    const screenshotButton = BABYLON.GUI.Button.CreateSimpleButton("button", "Screenshot");
    screenshotButton.width = "100px";
    screenshotButton.height = "40px";
    screenshotButton.color = "black";
    screenshotButton.background = "gray";
    screenshotButton.horizontalAlignment = BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_LEFT;
    screenshotButton.verticalAlignment = BABYLON.GUI.Control.VERTICAL_ALIGNMENT_TOP;
    screenshotButton.onPointerClickObservable.add(function() {
        for (const scene of scenes) {
            BABYLON.Tools.CreateScreenshotUsingRenderTarget(engine, scene.cameras[0], {precision: 1}, undefined, undefined, 10);
        };
    });

    advancedTexture.addControl(screenshotButton);

    engine.runRenderLoop(function() {
        for (let i = 0; i < views.length; i++) {
            if (views[i] === engine.activeView) {
                scenes[i].render();
            }
        };
    });
};