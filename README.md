# neural-surfaces

## Installation
To install with `pip`:
```
pip install git+https://github.com/vdorbs/neural-surfaces.git
```

To clone the repository and install dependencies:
```
git clone https://github.com/vdorbs/neural-surfaces.git
cd neural-surfaces
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Multi-GPU Systems
This library uses [`cholespy`](https://github.com/rgl-epfl/cholespy) to solve sparse linear systems with symmetric positive-definite matrices on CPU or GPU. On a multi-GPU system, you need to fool the solver into using anything other than `cuda:0`. For instance, suppose you want to run something on `cuda:2`. In your code, include:
```
from os import environ
environ['CUDA_VISIBLE_DEVICES'] = 2
```
or from the command line, run:
```
CUDA_VISIBLE_DEVICES=2 python your_script.py
```
## Oded Stein Meshes
Some of the meshes from [`odedstein-meshes`](https://github.com/odedstein/meshes) repository are available for download through the class `OdedSteinMeshes` in `neural_surfaces.utils`. As mentioned in that repository, you can cite that repository with the following `bibtex` snippet:
```
@article{odedstein-meshes,
  title={odedstein-meshes: A Computer Graphics Example Mesh Repository},
  author={Stein, Oded},
  notes={\url{odedstein.com/meshes}},
  year={2024}
}
```
but please be sure to credit the authors of each of the assets you use individually. Please make sure you follow the requirements of the respective licenses for each object in that collection.