# neural-surfaces

Tested with Python 3.10.14

To accelerate sparse linear solves with `umfpack`, use:
```
conda install suitesparse==5.10.1
pip install scikit-umfpack==0.3.0
```
and test in a Python environment with:
```
import scikits.umfpack
```
