'''

### B1 — Solving a Linear System

The following 4×4 linear system arises from a network flow problem:

```python
import numpy as np

A = np.array([[ 4., -1.,  0.,  1.],
              [-1.,  4., -1.,  0.],
              [ 0., -1.,  4., -1.],
              [ 1.,  0., -1.,  3.]], dtype=float)

b = np.array([7., 3., 8., 5.], dtype=float)
```

Use `scipy.linalg.solve` to find the vector `x` such that `Ax = b`. After computing `x`,
verify your result by computing the residual `r = b - A @ x` and 
reporting its infinity norm. A correct solution should give `||r||_∞ < 1e-12`.

result:

The solution to Ax = b.
x: [1.70714286 1.92857143 3.00714286 2.1       ]
residuals: [ 0.00000000e+00  8.88178420e-16  1.77635684e-15 -8.88178420e-16]
infinity norm: 1.7763568394002505e-15

'''

import numpy as np
from scipy.linalg import solve

A = np.array([[ 4., -1.,  0.,  1.],
              [-1.,  4., -1.,  0.],
              [ 0., -1.,  4., -1.],
              [ 1.,  0., -1.,  3.]], dtype=float)

b = np.array([7., 3., 8., 5.], dtype=float)

x = solve(A, b)

residual_vector = b - A.dot(x)

# infinity_norm = residual_vector.max()
infinity_norm = np.max(np.abs(residual_vector)) # don't overlook large negative with small positive

print(f' \nThe solution to Ax = b.\nx: {x}\nresiduals: {residual_vector}\ninfinity norm: {infinity_norm}')

apple = 1