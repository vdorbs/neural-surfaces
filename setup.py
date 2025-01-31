from setuptools import find_packages, setup


setup(
    name='neural-surfaces',
    packages=find_packages(),
    install_requires=[
        'cholespy',
        'cvxpy',
        'matplotlib',
        'mpmath',
        'polyscope',
        'potpourri3d',
        'pygeodesic',
        'torch',
        'torchsparsegradutils',
        'tqdm',
        'trimesh',
        'wandb'
    ],
    include_package_data=True,
    package_data={
        "neural_surfaces": ["render.js"]
    }
)