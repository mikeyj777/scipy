'''

### B4 — Eigendecomposition of a Symmetric Matrix

The following 4×4 matrix is symmetric positive definite:

```python
A = np.array([[5., 2., 1., 0.],
              [2., 6., 2., 1.],
              [1., 2., 7., 2.],
              [0., 1., 2., 5.]], dtype=float)
```

Use `scipy.linalg.eigh` (not `eig`) to compute the eigenvalues and eigenvectors. Your solution must:
1. Report the eigenvalues in ascending order (note what `eigh` gives you by default).
2. Verify orthonormality: confirm `Q.T @ Q ≈ I` to within `1e-12`.
3. Verify reconstruction: confirm `Q @ np.diag(lam) @ Q.T ≈ A` to within `1e-10`.

In a comment, explain why `eigh` is preferred over `eig` for symmetric matrices.

-----

1.  neigenvalues: [ 3.1917585   3.77709414  5.77421308 10.25693428]
    eigenvectors: [[ 0.68859897 -0.17984077  0.6153259  -0.33891413]
    [-0.56512447  0.41342261  0.44633794 -0.55722395]
    [-0.1149043  -0.6069169  -0.41628251 -0.66720143]
    [ 0.43961665  0.65451579 -0.49886406 -0.35983459]]


2.  Verify orthonormality: Q.T @ Q ≈ I: True


3. Verify reconstruction: Q @ np.diag(lam) @ Q.T ≈ A: True

"eigh" is preferred for symmetric matrices as it assumes symmetry, automatically pulling from the lower of the matrix.  


'''
import numpy as np
from scipy.linalg import eigh

A = np.array([[5., 2., 1., 0.],
              [2., 6., 2., 1.],
              [1., 2., 7., 2.],
              [0., 1., 2., 5.]], dtype=float)

w, v = eigh(A)

print(f'\n1. eigenvalues: {w}\neigenvectors: {v}')

Q = np.array(v)
lam = np.diag(w)
ident = np.identity(v.shape[0])

print(f'\n\n2.  Verify orthonormality: Q.T @ Q ≈ I: {np.allclose(Q.T @ v, ident, atol=1e-12)}')

print(f'\n\n3. Verify reconstruction: Q @ np.diag(lam) @ Q.T ≈ A: {np.allclose(Q @ lam @ Q.T, A, atol=1e-10)}')

apple = 1

