'''


### B2 — LU Factorization for Multiple Right-Hand Sides

Use the same matrix `A` from **B1**. You now need to solve the system for three different right-hand side vectors:

```python
b1 = np.array([1., 0., 0., 0.], dtype=float)
b2 = np.array([0., 1., 0., 0.], dtype=float)
b3 = np.array([2., -1., 3., 1.], dtype=float)
```

Use `scipy.linalg.lu_factor` to factor `A` **once**, then call `scipy.linalg.lu_solve` three times to recover `x1`, `x2`, and `x3`. 
Print each solution. Explain in a comment why this approach is more efficient than calling `scipy.linalg.solve` three separate times.

---

result:

b1 solution: [ 0.29285714  0.07142857 -0.00714286 -0.1       ]
b2 solution: [0.07142857 0.28571429 0.07142857 0.        ]
b3 solution: [0.39285714 0.07142857 0.89285714 0.5       ]

LU factorization is more efficient as it takes advantage of the positioning of 1s on the diagonal and upper/lower 0s 
to reduce complexity in matrix solution.

note - looking at the answer key.  A factorization is O(n3), which would be needed 3 times if using linalg.solve for b1-3.  
the factorization is only needed once with LU factorization.  
the operations to find the intermediate and x are O(n2), much less intensive.

'''

import numpy as np
from scipy.linalg import lu_factor, lu_solve

A = np.array([[ 4., -1.,  0.,  1.],
              [-1.,  4., -1.,  0.],
              [ 0., -1.,  4., -1.],
              [ 1.,  0., -1.,  3.]], dtype=float)

lu, piv = lu_factor(A)

lu_and_piv = (lu, piv)

b1 = np.array([1., 0., 0., 0.], dtype=float)
b2 = np.array([0., 1., 0., 0.], dtype=float)
b3 = np.array([2., -1., 3., 1.], dtype=float)

bs = [b1, b2, b3]
keys = ['b1', 'b2', 'b3']
ans = {}

for key, b in zip(keys, bs):
  try:
    ans[key] = lu_solve(lu_and_piv, b)
  except:
    pass

print(' \n')

for k, v in ans.items():
  print(f'{k} solution: {v}')