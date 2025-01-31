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